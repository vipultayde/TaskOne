// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    // Store parameters from multiple users
    mapping(address => mapping(string => int256[])) public userWeights;
    mapping(address => mapping(string => int256[])) public userBiases;

    // Averaged parameters
    mapping(string => int256[]) public averagedWeights;
    mapping(string => int256[]) public averagedBiases;

    // Track users who submitted parameters
    address[] public participants;
    mapping(address => bool) public hasSubmitted;

    // Track submission count for averaging
    uint256 public submissionCount;

    event ParametersSubmitted(address user, string layer);
    event ParametersAveraged(string layer, uint256 participantCount);

    constructor() {}

    function submitWeights(
        string memory layer,
        int256[] memory newWeights
    ) public {
        userWeights[msg.sender][layer] = newWeights;
        if (!hasSubmitted[msg.sender]) {
            participants.push(msg.sender);
            hasSubmitted[msg.sender] = true;
        }
        emit ParametersSubmitted(msg.sender, layer);
    }

    function submitBiases(
        string memory layer,
        int256[] memory newBiases
    ) public {
        userBiases[msg.sender][layer] = newBiases;
        if (!hasSubmitted[msg.sender]) {
            participants.push(msg.sender);
            hasSubmitted[msg.sender] = true;
        }
        emit ParametersSubmitted(msg.sender, layer);
    }

    // Batch submission function for all parameters in one transaction
    function submitAllParameters(
        string[] memory layers,
        int256[][] memory weights,
        int256[][] memory biases
    ) public {
        require(
            layers.length == weights.length && layers.length == biases.length,
            "Array lengths must match"
        );

        for (uint256 i = 0; i < layers.length; i++) {
            userWeights[msg.sender][layers[i]] = weights[i];
            userBiases[msg.sender][layers[i]] = biases[i];
        }

        if (!hasSubmitted[msg.sender]) {
            participants.push(msg.sender);
            hasSubmitted[msg.sender] = true;
        }

        emit ParametersSubmitted(msg.sender, "batch");
    }

    function averageParameters(string memory layer) public {
        require(participants.length > 0, "No participants yet");

        uint256 numParticipants = participants.length;
        uint256 paramLength;

        // Get parameter length from first participant
        if (userWeights[participants[0]][layer].length > 0) {
            paramLength = userWeights[participants[0]][layer].length;
            int256[] memory sumWeights = new int256[](paramLength);

            // Sum weights from all participants
            for (uint256 i = 0; i < numParticipants; i++) {
                int256[] memory userW = userWeights[participants[i]][layer];
                for (uint256 j = 0; j < paramLength; j++) {
                    sumWeights[j] += userW[j];
                }
            }

            // Average weights
            int256[] memory avgWeights = new int256[](paramLength);
            for (uint256 j = 0; j < paramLength; j++) {
                avgWeights[j] = sumWeights[j] / int256(numParticipants);
            }
            averagedWeights[layer] = avgWeights;
        }

        // Same for biases
        if (userBiases[participants[0]][layer].length > 0) {
            paramLength = userBiases[participants[0]][layer].length;
            int256[] memory sumBiases = new int256[](paramLength);

            for (uint256 i = 0; i < numParticipants; i++) {
                int256[] memory userB = userBiases[participants[i]][layer];
                for (uint256 j = 0; j < paramLength; j++) {
                    sumBiases[j] += userB[j];
                }
            }

            int256[] memory avgBiases = new int256[](paramLength);
            for (uint256 j = 0; j < paramLength; j++) {
                avgBiases[j] = sumBiases[j] / int256(numParticipants);
            }
            averagedBiases[layer] = avgBiases;
        }

        submissionCount = numParticipants;
        emit ParametersAveraged(layer, numParticipants);
    }

    function getAveragedWeights(
        string memory layer
    ) public view returns (int256[] memory) {
        return averagedWeights[layer];
    }

    function getAveragedBiases(
        string memory layer
    ) public view returns (int256[] memory) {
        return averagedBiases[layer];
    }

    function getUserWeights(
        address user,
        string memory layer
    ) public view returns (int256[] memory) {
        return userWeights[user][layer];
    }

    function getUserBiases(
        address user,
        string memory layer
    ) public view returns (int256[] memory) {
        return userBiases[user][layer];
    }

    function getParticipantCount() public view returns (uint256) {
        return participants.length;
    }

    // Prediction function using averaged parameters
    function predict(int256[] memory input) public view returns (int256) {
        // Scale factor for fixed-point arithmetic
        int256 SCALE = 1000000;

        // FC1: 8 -> 16
        int256[] memory fc1_weights = averagedWeights["fc1"];
        int256[] memory fc1_biases = averagedBiases["fc1"];
        require(fc1_weights.length == 8 * 16, "Invalid fc1 weights");
        require(fc1_biases.length == 16, "Invalid fc1 biases");

        int256[] memory layer1 = new int256[](16);
        for (uint i = 0; i < 16; i++) {
            int256 sum = 0;
            for (uint j = 0; j < 8; j++) {
                sum += (input[j] * fc1_weights[i * 8 + j]) / SCALE;
            }
            sum += fc1_biases[i];
            // ReLU
            layer1[i] = sum > 0 ? sum : 0;
        }

        // FC2: 16 -> 8
        int256[] memory fc2_weights = averagedWeights["fc2"];
        int256[] memory fc2_biases = averagedBiases["fc2"];
        require(fc2_weights.length == 16 * 8, "Invalid fc2 weights");
        require(fc2_biases.length == 8, "Invalid fc2 biases");

        int256[] memory layer2 = new int256[](8);
        for (uint i = 0; i < 8; i++) {
            int256 sum = 0;
            for (uint j = 0; j < 16; j++) {
                sum += (layer1[j] * fc2_weights[i * 16 + j]) / SCALE;
            }
            sum += fc2_biases[i];
            // ReLU
            layer2[i] = sum > 0 ? sum : 0;
        }

        // FC3: 8 -> 1
        int256[] memory fc3_weights = averagedWeights["fc3"];
        int256[] memory fc3_biases = averagedBiases["fc3"];
        require(fc3_weights.length == 8, "Invalid fc3 weights");
        require(fc3_biases.length == 1, "Invalid fc3 biases");

        int256 output = fc3_biases[0];
        for (uint i = 0; i < 8; i++) {
            output += (layer2[i] * fc3_weights[i]) / SCALE;
        }

        // Sigmoid approximation (simple)
        if (output > 500000) return SCALE; // ~1.0
        if (output < -500000) return 0; // ~0.0
        return (SCALE / 2) + (output / 2); // Linear approximation around 0.5
    }
}
