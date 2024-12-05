#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <netdb.h>
#include <fcntl.h>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

volatile sig_atomic_t terminate_requested = 0; // global flag for signal handling
// called async, so we need to make sure the compiler doesn't change it

// graceful termination
void signal_handler(int signum) {
    terminate_requested = 1;
}


struct RPCServer { // rpc server endpoint
    std::string address;
    std::string ip;
    int port;
    bool available;

    RPCServer(const std::string& addr) : // constructor to initialize and parse addr
        address(addr),
        available(true) {
        parse_address();
    }

    void parse_address() { // parse addr into IP and port
        size_t colon_pos = address.find(':');
        if (colon_pos != std::string::npos) {
            ip = address.substr(0, colon_pos);
            port = std::stoi(address.substr(colon_pos + 1));
        } else {
            ip = address;
            port = 50053; // RPC port for the PIs
        }
    }
};

class DurableLLaMA {
private:
    std::vector<RPCServer> servers; // rpc servers in the cluster
    std::vector<std::string> original_args; // cli arrguments from llama-cli
    pid_t llama_process; // PID of llamacpp
    std::mutex mtx; // thread mutex
    bool should_continue; // control flag for continue loop
    int original_ngl; // gpu layers from llama-cli
    int stdout_pipe[2];
    int stderr_pipe[2];
    std::chrono::steady_clock::time_point last_output_time; // last output timer

    std::string build_rpc_string() { // string of available RPC servers
        std::string rpc_servers;
        bool first = true;
        for (auto& server : servers) { // for all servers
            if (server.available) { // check if available
                if (!first) rpc_servers += ","; // separate them
                rpc_servers += server.address; //
                first = false;
            }
        }
        return rpc_servers;
    }
    // builts to format "ip1:port1,ip2:port2, etc"

    bool is_server_reachable(const std::string& ip, int port) { // open a TCP connection to see if server is reachable
        int sockfd;
        struct sockaddr_in serv_addr;
        sockfd = socket(AF_INET, SOCK_STREAM, 0); // ipv4, tcp
        if (sockfd < 0) { // check if socket creation failed
            perror("socket");
            return false;
        }

        memset(&serv_addr, 0, sizeof(serv_addr)); // server address setup stuff
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port); // convert port to network byte order

        if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) { // IP is convered to binary
            close(sockfd);
            return false;
        }

        struct timeval tv;
        tv.tv_sec = 5; // 5-second timeout
        tv.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv); // set socket options for send and receive timeout
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);

        int result = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); // connect attempt
        close(sockfd);

        return result == 0; // return true if connection succeeded
    }

    // inference status check
    void check_inference_status() {
        auto now = std::chrono::steady_clock::now(); // get current time
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - last_output_time).count(); // calc time since last output

        if (elapsed >= 5) { // restarts inference on remaining PIs if no server is available
            std::cout << "\nNo output received for 5 seconds, attempting restart..." << std::endl;

            bool any_server_removed = false; // initialized server removal flag to false
            for (auto& server : servers) {
                if (server.available && !is_server_reachable(server.ip, server.port)) { // if server is marked available but can't be reached
                    server.available = false;
                    any_server_removed = true;
                    std::cout << "Removing unreachable server " << server.address << " and trying again..." << std::endl;
                    // mark unavailable, set removal flag, log removal
                }
            }

            if (!any_server_removed) {
                std::cout << "All RPC servers are reachable, but no output received. Restarting inference..." << std::endl; // for stalled inference
            } else if (std::none_of(servers.begin(), servers.end(), [](const RPCServer& s){ return s.available; })) { // if all servers are unavailable
                std::cout << "No reachable RPC servers available, falling back to CPU..." << std::endl; // fallback to cpu
            }

            restart_llama();
        }
    }

    std::vector<char*> build_command_args() { //extracts and rebuilds command line args from llama.cpp
        std::vector<char*> args; // vector for cli arguments
        args.push_back(strdup("./llama-cli")); // add exec name for first arg

        bool skip_next = false; // skip args
        std::string rpc_servers = build_rpc_string(); //get string of available servers

        // Check if we have any available RPC servers, if not, fallback to CPU only
        bool is_fallback = rpc_servers.empty();

        for (size_t i = 0; i < original_args.size(); i++) { // process all llama-cli arguments except RPC and NGL
            if (skip_next) {
                skip_next = false;
                continue;
            }

            // Skip existing RPC and ngl arguments
            if (original_args[i] == "--rpc" ||
                original_args[i] == "-ngl" ||
                original_args[i] == "--n-gpu-layers") {
                skip_next = true;
                continue;
            }

            args.push_back(strdup(original_args[i].c_str())); // add args to vector
        }

        if (!is_fallback) { // add RPC and ngl arguments from command line
            args.push_back(strdup("--rpc"));
            args.push_back(strdup(rpc_servers.c_str()));
            args.push_back(strdup("-ngl"));
            args.push_back(strdup(std::to_string(original_ngl).c_str()));
        } else {
            args.push_back(strdup("-ngl"));
            args.push_back(strdup("0"));  // if all RPC servers fail
        }

        args.push_back(nullptr); // null term for execv
        return args;
    }

    void restart_llama() {
        if (llama_process > 0) {
            kill(llama_process, SIGTERM); // kill llama.cli if it's already running
            int status;
            waitpid(llama_process, &status, 0);
        }

        if (stdout_pipe[0] != -1) close(stdout_pipe[0]); // pipe cleaning and reinstantiation
        if (stdout_pipe[1] != -1) close(stdout_pipe[1]);

        pipe(stdout_pipe);

        auto args = build_command_args(); // build argument array for new process

        llama_process = fork(); // fork and execute llama-cli binary
        if (llama_process == 0) { // if code runs in child proccess
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stdout_pipe[1], STDERR_FILENO); // redirect stderr to stdout
            close(stdout_pipe[0]);
            close(stdout_pipe[1]); //close pipe

            execv("./llama-cli", args.data()); //replace child process with llama-cli
            perror("execv failed");
            exit(1);
        }

        close(stdout_pipe[1]); //non-blocking IO stuff

        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);

        for (char* arg : args) { // free up arg memory
            if (arg) free(arg);
        }

        last_output_time = std::chrono::steady_clock::now(); // check if inference is still working
    }


    void monitor_output() { // read and handle output from llama-cli process
        fd_set read_fds;
        struct timeval tv;
        int retval; //file descriptor set, timeout struct, and return value for select()

        FD_ZERO(&read_fds);
        FD_SET(stdout_pipe[0], &read_fds); // init file descriptor sedd and add pipe read end


        tv.tv_sec = 1;
        tv.tv_usec = 0;

        retval = select(stdout_pipe[0] + 1, &read_fds, NULL, NULL, &tv); // wait for data pipe

        if (retval == -1) {
            perror("select()"); // error handling
        } else if (retval > 0) { //read and display output from llama-cli's inference engine
            if (FD_ISSET(stdout_pipe[0], &read_fds)) {
                char buffer[4096];
                ssize_t n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1); // create buffer and read tokens from pipe
                if (n > 0) {
                    buffer[n] = '\0'; // if data was read, null term string
                    std::cout << buffer << std::flush;
                    last_output_time = std::chrono::steady_clock::now(); // ouput toks and update output timestamp
                }
            }
        }
    }


    int find_ngl_value() { // get gpu layers from CLI arguments
        for (size_t i = 0; i < original_args.size() - 1; i++) {
            if (original_args[i] == "-ngl" || original_args[i] == "--n-gpu-layers") {
                return std::stoi(original_args[i + 1]);
            }
        }
        return 99;  // offload all layers by default
    }

public:
    DurableLLaMA(const std::vector<std::string>& server_addresses, int argc, char** argv) // constructor for server addresses andd CLI args
        : llama_process(-1), // start with invalid process ID
          should_continue(true) { // set continue flag to true

        for (const auto& addr : server_addresses) { // create rpc server objects for each address and add to server vectors
            servers.emplace_back(addr); // construct object in place
        }

        for (int i = 1; i < argc; i++) { // store command line args
            original_args.push_back(argv[i]);
        }

        original_ngl = find_ngl_value(); // get gpu layers
        stdout_pipe[0] = stdout_pipe[1] = -1;
        stderr_pipe[0] = stderr_pipe[1] = -1; // initialize pipe descriptors (no valid fd)
        last_output_time = std::chrono::steady_clock::now(); // initialize timestamp to current time
    }

    void run() {
        restart_llama(); // start llama-cli process
        while (should_continue && !terminate_requested) { // loop until terminated
            monitor_output();
            check_inference_status(); // see if inference is still running
            int status; // see if process is terminated
            pid_t result = waitpid(llama_process, &status, WNOHANG); // non-blocking check

            if (result == llama_process) { // if terminated
                if (WIFEXITED(status)) {
                    int exit_status = WEXITSTATUS(status);
                    std::cout << "LLaMA process exited with status " << exit_status << "." << std::endl;
                    if (exit_status == 0) {
                        // Inference completed successfully
                        should_continue = false;
                    } else {
                        // Non-zero exit status, restart
                        std::cout << "LLaMA process exited with non-zero status. Restarting..." << std::endl;
                        restart_llama();
                    }
                } else if (WIFSIGNALED(status)) {
                    std::cout << "LLaMA process was terminated by a signal. Restarting..." << std::endl;
                    restart_llama();
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Clean up before exiting
        if (llama_process > 0) {
            kill(llama_process, SIGTERM);
            int status;
            waitpid(llama_process, &status, 0);
        }
    }
};

int main(int argc, char** argv) { // for command line arguments
    std::vector<std::string> rpc_servers; // store rpc server addresses in a vector

    // Register signal handler for graceful termination
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    // handlers for interrupt signals

    for (int i = 1; i < argc - 1; i++) { // parse server addresses from command line
        if (strcmp(argv[i], "--rpc") == 0) {
            std::string servers = argv[i + 1];
            size_t pos = 0; // get server string and position
            while ((pos = servers.find(',')) != std::string::npos) { // while there are still RPC servers
                rpc_servers.push_back(servers.substr(0, pos));
                servers.erase(0, pos + 1); // extract addresses and remove from string
            }
            rpc_servers.push_back(servers); // add final server address
            break;
        }
    }

    if (rpc_servers.empty()) {
        std::cerr << "Usage: " << argv[0] << " [llama.cpp options] --rpc server1:port1,server2:port2,...\n";
        return 1;
    }

    DurableLLaMA llama(rpc_servers, argc, argv); //create and run wrapper
    llama.run();

    return 0;
}