**Overview**<br>
This repository provides a Python SDK that acts as a wrapper around Forte’s O2 Oracle API and Rules Engine, making blockchain integration seamless for data scientists and ML engineers — no blockchain or JavaScript expertise required.

*The SDK allows users to*:<br>

Submit model metrics and metadata to the blockchain via simple Python dictionaries.<br>

Automatically serialize this data to local JSON files, which are then read by JavaScript modules that interact with Forte’s APIs.<br>

Define and manage smart contract logic for model promotion, monitoring, and governance through JSON-based rules compatible with Forte’s Rules Engine.<br>

This SDK is designed to be embedded directly into existing ML workflows:<br>

Easily integrate into training, promotion, and monitoring pipelines.<br>

Works in standard Python environments (cloud platforms, local dev setups, or Jupyter notebooks).<br>

No code migration or environment changes are needed — continue running ML tasks in pure Python.<br>

[ML Universe UI](docs/ML_Universe.png)

[Link to the Whitepaper](docs/Decentralized_MLOps_Forte.pdf)

[Watch Demo Video](https://drive.google.com/file/d/your-file-id/view)

[Link to our presentation in Canva](https://www.canva.com/design/DAGnhlJu3RY/EDm30vFwvf9E6uZmZzSOBw/edit?utm_content=DAGnhlJu3RY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

**Project Structure**<br>
<pre>
.
├── BlockchainWrapper.py     # Main Python SDK – import this in ML code
├── ML_Models/
│   ├── model_1/
│   │   ├── train.py
│   │   └── monitor.py
│   ├── model_2/
│   │   ├── train.py
│   │   └── monitor.py
│   └── ...
├── ML_Verse/
│   └── ...                  # Low-level JavaScript modules for Forte API calls
├── scripts/
│   └── ...                  # (Optional) utility or execution scripts
├── docs/
│   └── ...                  # Diagrams, documentation, media assets
├── README.md
└── requirements.txt
</pre>

**Requirements**<br>
Python and Node JS are required to interact with this library. Node JS is needed on the backend to run the abstracted Javascript code under *ML_Verse* with Forte API calls.<br>

**Key Components**<br>
*BlockchainWrapper.py* – Python SDK (Main Entry Point)<br>
This is the core module users will interact with. It provides a simple, Pythonic interface to:<br>

Validate and serialize ML metrics to JSON.<br>

Route metrics to the correct destination (train, monitor, etc.) based on the ml_step argument and Python dictionaries.<br>

Trigger the underlying JS modules to push data to the Forte blockchain via the O2 Oracle API.<br>

Usage:<br>
Import this file into your ML training or monitoring scripts, pass your metrics as a dictionary, and specify the ml_step you're in ('train', 'monitor', 'promote', etc.). The wrapper handles the rest.<br>

*ML_Models/* – Example ML Workflows<br>
This folder provides end-to-end examples of how to use the wrapper in real-world MLOps scenarios:<br>

Each subfolder represents a different ML model.<br>

Each contains:<br>

train.py: Simulated training flow that logs metrics to the blockchain.<br>

monitor.py: Example monitoring flow for tracking post-deployment metrics.<br>

The goal is to show users how to integrate the SDK into any ML pipeline with minimal changes. Just import the SDK, structure your metrics as dictionaries, and call the appropriate method with the ml_step to add the metrics with timestamps to the chain.<br>

*ML_Verse/* – Internal JavaScript Modules (No User Action Needed)<br>
This directory contains low-level JavaScript code that interfaces with Forte’s APIs. These modules:<br>

Handle actual HTTP requests to Forte’s O2 Oracle and Rules Engine.<br>

Read from the JSON files produced by the Python wrapper.<br>

🔒 Note: End users do not need to interact with this directory directly. All interactions are abstracted through BlockchainWrapper.py and it is just for the backend and stores temporary JSON serializations of metrics.

