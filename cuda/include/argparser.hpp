#ifndef ARGPARSER_HPP
#define ARGPARSER_HPP

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
// A simple command-line argument parser
class ArgParser {
public:
    ArgParser(const std::string& description = "") : description_(description) {};
    // Add an argument with a name, description, and default value
    template <typename T>
    void add_argument(const std::string& name, const std::string& description, const T& defaultValue) {
        if (args_.count(name)) {
            throw std::invalid_argument("Argument already exists: " + name);
        }
        args_[name] = Argument(description, toString(defaultValue));
    }

    // Parse command-line arguments
    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0) { // Check if it starts with "--"
                std::string name = arg.substr(2);
                if (args_.count(name)) {
                    if (i + 1 < argc) {
                        args_[name].value = argv[++i];
                    } else {
                        throw std::invalid_argument("No value provided for argument: " + name);
                    }
                } else {
                    throw std::invalid_argument("Unknown argument: " + name);
                }
            }
        }
    }

    // Get the value of an argument with the appropriate type
    template <typename T>
    T get(const std::string& name) const {
        if (!args_.count(name)) {
            throw std::invalid_argument("Argument not found: " + name);
        }
        return fromString<T>(args_.at(name).value);
    }

    // Print help message
    void print_help() const {
        std::cout << "Available arguments:\n";
        for (const auto& [name, arg] : args_) {
            std::cout << "--" << name << ": " << arg.description << " (default: " << arg.value << ")\n";
        }
    }

private:
    struct Argument {
        std::string description;
        std::string value;

        Argument() = default;  // Default constructor necessary to instantiate args_
        Argument(const std::string& desc, const std::string& val)
            : description(desc), value(val) {}
    };

    std::unordered_map<std::string, Argument> args_;
    std::string description_;
    // Convert a value to a string
    template <typename T>
    static std::string toString(const T& value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }

    // Convert a string to a value of type T
    template <typename T>
    static T fromString(const std::string& str) {
        std::istringstream iss(str);
        T value;
        if (!(iss >> value)) {
            throw std::invalid_argument("Invalid value: " + str);
        }
        return value;
    }
};

// Specializations for std::string
template <>
inline std::string ArgParser::fromString<std::string>(const std::string& str) {
    return str;
}

template <>
inline std::string ArgParser::toString<std::string>(const std::string& value) {
    return value;
}

#endif // ARGPARSER_HPP