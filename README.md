**Overview**
This repository provides a Python SDK that acts as a wrapper around Forte’s O2 Oracle API and Rules Engine, making blockchain integration seamless for data scientists and ML engineers — no blockchain or JavaScript expertise required.

*The SDK allows users to*:

Submit model metrics and metadata to the blockchain via simple Python dictionaries.

Automatically serialize this data to local JSON files, which are then read by JavaScript modules that interact with Forte’s APIs.

Define and manage smart contract logic for model promotion, monitoring, and governance through JSON-based rules compatible with Forte’s Rules Engine.

This SDK is designed to be embedded directly into existing ML workflows:

Easily integrate into training, promotion, and monitoring pipelines.

Works in standard Python environments (cloud platforms, local dev setups, or Jupyter notebooks).

No code migration or environment changes are needed — continue running ML tasks in pure Python.

![ML Universe UI](https://drive.google.com/file/d/1vFoOssWfRkdpq3DM0ECy16n0K-uAan5R/view?usp=sharing)

[Watch Demo Video](https://drive.google.com/file/d/your-file-id/view)

[Link to our presentation in Canva](https://www.canva.com/design/DAGnhlJu3RY/EDm30vFwvf9E6uZmZzSOBw/edit?utm_content=DAGnhlJu3RY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

**Project Structure**
<pre> ```
.
├── BlockchainWrapper.py     # Main Python SDK – import this in ML code
├── ML_Models/
│   ├── model_a/
│   │   ├── train.py
│   │   └── monitor.py
│   ├── model_b/
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
``` </pre>

**Requirements**
Python and Node JS are required to interact with this library. Node JS is needed on the backend to run the abstracted Javascript code under *ML_Verse* with Forte API calls.

**Key Components**
*BlockchainWrapper.py* – Python SDK (Main Entry Point)
This is the core module users will interact with. It provides a simple, Pythonic interface to:

Validate and serialize ML metrics to JSON.

Route metrics to the correct destination (train, monitor, etc.) based on the ml_step argument and Python dictionaries.

Trigger the underlying JS modules to push data to the Forte blockchain via the O2 Oracle API.

Usage:
Import this file into your ML training or monitoring scripts, pass your metrics as a dictionary, and specify the ml_step you're in ('train', 'monitor', 'promote', etc.). The wrapper handles the rest.

*ML_Models/* – Example ML Workflows
This folder provides end-to-end examples of how to use the wrapper in real-world MLOps scenarios:

Each subfolder represents a different ML model.

Each contains:

train.py: Simulated training flow that logs metrics to the blockchain.

monitor.py: Example monitoring flow for tracking post-deployment metrics.

The goal is to show users how to integrate the SDK into any ML pipeline with minimal changes. Just import the SDK, structure your metrics as dictionaries, and call the appropriate method with the ml_step to add the metrics with timestamps to the chain.

*ML_Verse/* – Internal JavaScript Modules (No User Action Needed)
This directory contains low-level JavaScript code that interfaces with Forte’s APIs. These modules:

Handle actual HTTP requests to Forte’s O2 Oracle and Rules Engine.

Read from the JSON files produced by the Python wrapper.

🔒 Note: End users do not need to interact with this directory directly. All interactions are abstracted through BlockchainWrapper.py and it is just for the backend and stores temporary JSON serializations of metrics.