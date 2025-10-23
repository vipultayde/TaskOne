// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {
    // Store model hashes from users (off-chain storage of actual parameters)
    mapping(address => string) public modelHashes;

    // Track users who submitted models
    address[] public participants;
    mapping(address => bool) public hasSubmitted;

    // Track submission count
    uint256 public submissionCount;

    // Simple prediction result storage (for demo purposes)
    mapping(address => uint256) public predictions;

    event ModelSubmitted(address user, string modelHash);
    event PredictionMade(address user, uint256 prediction);

    constructor() {}

    // Submit model hash (actual model parameters stored off-chain)
    function submitModel(string memory modelHash) public {
        modelHashes[msg.sender] = modelHash;
        if (!hasSubmitted[msg.sender]) {
            participants.push(msg.sender);
            hasSubmitted[msg.sender] = true;
        }
        submissionCount = participants.length;
        emit ModelSubmitted(msg.sender, modelHash);
    }

    // Store prediction result (called by off-chain process)
    function storePrediction(uint256 prediction) public {
        predictions[msg.sender] = prediction;
        emit PredictionMade(msg.sender, prediction);
    }

    function getParticipantCount() public view returns (uint256) {
        return participants.length;
    }

    function getModelHash(address user) public view returns (string memory) {
        return modelHashes[user];
    }

    function getPrediction(address user) public view returns (uint256) {
        return predictions[user];
    }
}
