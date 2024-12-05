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

using std::string;
using std::to_string;
using std::stoi;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::mutex;
using std::chrono;

// global flag for signal handling
// called async, so we need to make sure the compiler doesn't change it
volatile sig_atomic_t terminateRequested = 0;

// graceful termination, takes signals like sigterm, sigkill
void signalHandler(int signum) {
    terminateRequested = 1;
}

// grabs the rpc server address and port from command line
struct RPCServer {
    string address;
    string ip;
    int port;
    bool available;

    RPCServer(const string& addr) : // constructor to initialize and parse addr
        address(addr),
        available(true) {
        parseAddress();
    }

    void parseAddress() { // parse addr into IP and port
        size_t colon_pos = address.find(':');
        if (colon_pos != string::npos) {
            ip = address.substr(0, colon_pos);
            port = stoi(address.substr(colon_pos + 1));
        } else {
            ip = address;
            port = 50053; // RPC port for the PIs
        }
    }
};

class DurableLLaMA {
private:
    vector<RPCServer> servers; // rpc servers in the cluster
    vector<string> originalArgs; // cli arrguments from llama-cli
    pid_t llamaProcess; // PID of llamacpp
    mutex mtx; // thread mutex
    bool shouldContinue; // control flag for continue loop
    int original_ngl; // gpu layers from llama-cli
    int stdout_pipe[2];
    int stderr_pipe[2];
    chrono::steady_clock::time_point lastOutputTime; // last output timer

    string buildRPCstring() { // string of available RPC servers
        string rpcServers;
        bool first = true;
        for (auto& server : servers) {
            if (server.available) {
                if (!first) rpcServers += ",";
                rpcServers += server.address;
                first = false;
            }
        }
        return rpcServers;
    }

    bool isServerReachable(const string& ip, int port) { // open a TCP connection to see if server is reachable
        int sockfd;
        struct sockaddr_in serv_addr;
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("socket");
            return false;
        }

        memset(&serv_addr, 0, sizeof(serv_addr)); // server address setup stuff
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) { // IP is convered to binary
            close(sockfd);
            return false;
        }

        struct timeval tv;
        tv.tv_sec = 5; // 5-second timeout
        tv.tv_usec = 0;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
        setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);

        int result = connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); // connect attempt
        close(sockfd);

        return result == 0;
    }

    // inference status check
    void checkInferenceStatus() {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::seconds>(
            now - lastOutputTime).count();

        if (elapsed >= 5) { // restarts inference on remaining PIs if no server is available
            cout << "\nNo output received for 5 seconds, attempting restart..." << endl;

            bool anyServerRemoved = false;
            for (auto& server : servers) {
                if (server.available && !isServerReachable(server.ip, server.port)) {
                    server.available = false;
                    anyServerRemoved = true;
                    cout << "Removing unreachable server " << server.address << " and trying again..." << endl;
                }
            }

            // condition where haven't received output in 5 seconds, but servers are still reachable
            // could mean that servers are in a bad state, responding to TCP connections, but
            // not processing inference requests properly
            if (!anyServerRemoved) {
                cout << "All RPC servers are reachable, but no output received. Restarting inference..." << endl;
            } else {
                // Check if all servers are unavailable
                bool allServersDown = true;
                for (const auto& server : servers) {
                    if (server.available) {
                        allServersDown = false;   // still have a server available
                        break;
                    }
                }
                if (allServersDown) {
                    cout << "No reachable RPC servers available, falling back to CPU..." << endl;
                }
            }

            restartLlama();
        }
    }

    vector<char*> buildCommandArgs() { //extracts and rebuilds command line args from llama.cpp
        vector<char*> args;
        args.push_back(strdup("./llama-cli"));

        bool skipNext = false;
        string rpcServers = buildRPCstring();

        // Check if we have any available RPC servers, if not, fallback to CPU only
        bool isFallback = rpcServers.empty();

        for (size_t i = 0; i < originalArgs.size(); i++) { // process all llama-cli arguments except RPC and NGL
            if (skipNext) {
                skipNext = false;
                continue;
            }

            // Skip existing RPC and ngl arguments
            if (originalArgs[i] == "--rpc" ||
                originalArgs[i] == "-ngl" ||
                originalArgs[i] == "--n-gpu-layers") {
                skipNext = true;
                continue;
            }

            args.push_back(strdup(originalArgs[i].c_str()));
        }

        if (!isFallback) { // add RPC and ngl arguments from command line
            args.push_back(strdup("--rpc"));
            args.push_back(strdup(rpcServers.c_str()));
            args.push_back(strdup("-ngl"));
            args.push_back(strdup(to_string(original_ngl).c_str()));
        } else {
            args.push_back(strdup("-ngl"));
            args.push_back(strdup("0"));  // if all RPC servers fail
        }

        args.push_back(nullptr);
        return args;
    }

    void restartLlama() {
        if (llamaProcess > 0) {
            kill(llamaProcess, SIGTERM); // kill llama.cli if it's already running
            int status;
            waitpid(llamaProcess, &status, 0);
        }

        if (stdout_pipe[0] != -1) close(stdout_pipe[0]); // pipe cleaning and reinstantiation
        if (stdout_pipe[1] != -1) close(stdout_pipe[1]);

        pipe(stdout_pipe);

        auto args = buildCommandArgs();

        llamaProcess = fork(); // fork and execute llama-cli binary
        if (llamaProcess == 0) {
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stdout_pipe[1], STDERR_FILENO); // redirect stderr to stdout
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);

            execv("./llama-cli", args.data());
            perror("execv failed");
            exit(1);
        }

        close(stdout_pipe[1]); //non-blocking IO stuff

        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);

        for (char* arg : args) { // free up arg memory
            if (arg) free(arg);
        }

        lastOutputTime = chrono::steady_clock::now(); // check if inference is still working
    }

    void monitorOutput() {
        fd_set read_fds;
        struct timeval tv;
        int retval;

        FD_ZERO(&read_fds);
        FD_SET(stdout_pipe[0], &read_fds);

        // Set timeout to 1 second
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        retval = select(stdout_pipe[0] + 1, &read_fds, NULL, NULL, &tv);

        if (retval == -1) {
            perror("select()");
        } else if (retval > 0) { //read and display output from llama-cli's inference engine
            if (FD_ISSET(stdout_pipe[0], &read_fds)) {
                char buffer[4096];
                ssize_t n = read(stdout_pipe[0], buffer, sizeof(buffer) - 1);
                if (n > 0) {
                    buffer[n] = '\0';
                    cout << buffer << flush;
                    lastOutputTime = chrono::steady_clock::now();
                }
            }
        }
    }


    int findNglValue() { // get gpu layers from CLI arguments
        for (size_t i = 0; i < originalArgs.size() - 1; i++) {
            if (originalArgs[i] == "-ngl" || originalArgs[i] == "--n-gpu-layers") {
                return stoi(originalArgs[i + 1]);
            }
        }
        return 99;  // offload all layers by default
    }

public:
    DurableLLaMA(const vector<string>& server_addresses, int argc, char** argv) // constructor for server addresses andd CLI args
        : llamaProcess(-1),
          shouldContinue(true) {

        for (const auto& addr : server_addresses) {
            servers.emplace_back(addr);
        }

        for (int i = 1; i < argc; i++) {
            originalArgs.push_back(argv[i]);
        }

        original_ngl = findNglValue();
        stdout_pipe[0] = stdout_pipe[1] = -1;
        stderr_pipe[0] = stderr_pipe[1] = -1;
        lastOutputTime = chrono::steady_clock::now();
    }

    void run() {
        restartLlama();

        while (shouldContinue && !terminateRequested) {
            monitorOutput();
            checkInferenceStatus();

            int status; // see if process is terminated
            pid_t result = waitpid(llamaProcess, &status, WNOHANG);

            if (result == llamaProcess) {
                if (WIFEXITED(status)) {
                    int exit_status = WEXITSTATUS(status);
                    cout << "LLaMA process exited with status " << exit_status << "." << endl;
                    if (exit_status == 0) {
                        // Inference completed successfully
                        shouldContinue = false;
                    } else {
                        // Non-zero exit status, restart
                        cout << "LLaMA process exited with non-zero status. Restarting..." << endl;
                        restartLlama();
                    }
                } else if (WIFSIGNALED(status)) {
                    cout << "LLaMA process was terminated by a signal. Restarting..." << endl;
                    restartLlama();
                }
            }

            std::this_thread::sleep_for(chrono::milliseconds(100));
        }

        // Clean up before exiting
        if (llamaProcess > 0) {
            kill(llamaProcess, SIGTERM);
            int status;
            waitpid(llamaProcess, &status, 0);
        }
    }
};

int main(int argc, char** argv) {
    vector<string> rpcServers;

    // Register signal handler for graceful termination
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    for (int i = 1; i < argc - 1; i++) { // parse server addresses from command line
        if (strcmp(argv[i], "--rpc") == 0) {
            string servers = argv[i + 1];
            size_t pos = 0;
            while ((pos = servers.find(',')) != string::npos) {
                rpcServers.push_back(servers.substr(0, pos));
                servers.erase(0, pos + 1);
            }
            rpcServers.push_back(servers);
            break;
        }
    }

    if (rpcServers.empty()) {
        cerr << "Usage: " << argv[0] << " [llama.cpp options] --rpc server1:port1,server2:port2,...\n";
        return 1;
    }

    DurableLLaMA llama(rpcServers, argc, argv); //create and run wrapper
    llama.run();

    return 0;
}