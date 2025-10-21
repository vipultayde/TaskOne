// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ParameterStorage {
    mapping(string => int256[]) public weights;
    mapping(string => int256[]) public biases;
    address public owner;

    event ParametersUpdated(
        string layer,
        int256[] newWeights,
        int256[] newBiases
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can update parameters");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function updateWeights(
        string memory layer,
        int256[] memory newWeights
    ) public onlyOwner {
        weights[layer] = newWeights;
        emit ParametersUpdated(layer, newWeights, biases[layer]);
    }

    function updateBiases(
        string memory layer,
        int256[] memory newBiases
    ) public onlyOwner {
        biases[layer] = newBiases;
        emit ParametersUpdated(layer, weights[layer], newBiases);
    }

    function getWeights(
        string memory layer
    ) public view returns (int256[] memory) {
        return weights[layer];
    }

    function getBiases(
        string memory layer
    ) public view returns (int256[] memory) {
        return biases[layer];
    }
}
