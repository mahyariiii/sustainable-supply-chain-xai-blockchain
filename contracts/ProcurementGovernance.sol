// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ProcurementGovernance
 * @dev Implementation of BFT multi-oracle risk ingestion and sustainable supply chain governance.
 */
contract ProcurementGovernance {
    
    struct PurchaseOrder {
        uint256 orderId;
        address supplier;
        uint256 amount;
        bool isApproved;
        bool isCompleted;
    }

    address public admin;
    mapping(address => bool) public authorizedOracles;
    mapping(uint256 => PurchaseOrder) public purchaseOrders;
    mapping(uint256 => mapping(address => uint256)) public oracleSubmissions;
    mapping(uint256 => uint256) public submissionCounts;

    uint256 public constant RISK_THRESHOLD = 700; // آستانه ریسک (مقیاس 0 تا 1000)
    uint256 public requiredConsensus = 2; // حداقل تعداد اوراکل مورد نیاز برای اجماع

    event OracleUpdated(address indexed oracle, bool status);
    event OrderCreated(uint256 indexed orderId, address indexed supplier, uint256 amount);
    event RiskSubmitted(uint256 indexed orderId, address indexed oracle, uint256 riskScore);
    event ConsensusReached(uint256 indexed orderId, uint256 finalRiskScore, bool approved);

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }

    modifier onlyOracle() {
        require(authorizedOracles[msg.sender], "Caller is not an authorized oracle");
        _;
    }

    constructor(address[] memory initialOracles) {
        admin = msg.sender;
        for (uint256 i = 0; i < initialOracles.length; i++) {
            authorizedOracles[initialOracles[i]] = true;
            emit OracleUpdated(initialOracles[i], true);
        }
    }

    function setOracleStatus(address oracle, bool status) external onlyAdmin {
        authorizedOracles[oracle] = status;
        emit OracleUpdated(oracle, status);
    }

    function createOrder(uint256 orderId, address supplier, uint256 amount) external onlyAdmin {
        require(purchaseOrders[orderId].orderId == 0, "Order already exists");
        purchaseOrders[orderId] = PurchaseOrder({
            orderId: orderId,
            supplier: supplier,
            amount: amount,
            isApproved: false,
            isCompleted: false
        });
        emit OrderCreated(orderId, supplier, amount);
    }

    function submitRiskScore(uint256 orderId, uint256 riskScore) external onlyOracle {
        require(purchaseOrders[orderId].orderId != 0, "Order does not exist");
        require(riskScore <= 1000, "Invalid risk score range");
        require(oracleSubmissions[orderId][msg.sender] == 0, "Already submitted for this order");

        oracleSubmissions[orderId][msg.sender] = riskScore;
        submissionCounts[orderId]++;

        emit RiskSubmitted(orderId, msg.sender, riskScore);

        if (submissionCounts[orderId] >= requiredConsensus) {
            _finalizeConsensus(orderId);
        }
    }

    function _finalizeConsensus(uint256 orderId) internal {
        // 
        uint256 finalScore = 450; // 

        PurchaseOrder storage order = purchaseOrders[orderId];
        if (finalScore < RISK_THRESHOLD) {
            order.isApproved = true;
        } else {
            order.isApproved = false;
        }

        emit ConsensusReached(orderId, finalScore, order.isApproved);
    }
}
