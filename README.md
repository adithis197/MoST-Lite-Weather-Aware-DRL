README
======

Weather-Aware Autonomous Driving with TD3
-----------------------------------------

Reinforcement Learning using Multi-Modal Scene Embeddings from the Waymo Open Dataset

Overview
--------

This project implements a weather-aware driving policy trained using the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. The learning pipeline integrates visual features, LiDAR proximity information, vehicle dynamics, and weather context into a unified state representation.

The work is influenced by ideas from the MOST (Multi-Scene Tokenization) paper, which demonstrated that fusing scene representations from multiple sensors improves downstream learning performance in autonomous driving tasks.

Reference:MOST: Multi-Modal Scene Tokenization for Autonomous Drivinghttps://arxiv.org/abs/2406.09697

This project provides a lightweight version of multi-modal scene tokenization by combining:

*   ViT-B/16 image embeddings
    
*   LiDAR minimum distance features
    
*   CLIP cosine similarity for weather classification
    
*   One-hot encoded weather conditions
    
*   Synthetic expert driving actions
    
*   A TD3 reinforcement-learning agent
    

1\. Waymo Data
--------------

Waymo Open Dataset:[https://waymo.com/open/](https://waymo.com/open/?utm_source=chatgpt.com)

Download TFRecord files containing camera images, LiDAR, and calibration information.

Folder structure used in this project:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   waymo/      raw/      frames/      lidar/      metadata/      embeddings/      td3_buffer_final/   `

2\. Environment Setup
---------------------

The project uses Python 3.10, TensorFlow 2.11 (for Waymo), and PyTorch (for TD3).Run the following commands:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   sudo apt-get update  sudo apt-get install python3.10 python3.10-venv python3.10-distutils  python3.10 -m venv waymo310  source waymo310/bin/activate  pip install tensorflow==2.11.0  pip install waymo-open-dataset-tf-2-11-0  pip install torch torchvision torchaudio  pip install matplotlib imageio opencv-python   `

3\. Data Preprocessing Pipeline
-------------------------------

The preprocessing steps below must be completed in order. Each file name matches the scripts provided in this repository.

### Step 3.1 — Extract Camera Frames

Script: extract\_frames.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python extract_frames.py   `

Converts TFRecord segments into directory-based RGB frames for each Waymo scene. These frames are used to generate ViT embeddings.

### Step 3.2 — Extract LiDAR and Compute Minimum Distance

Script: extract\_lidar\_min\_dist.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python extract_lidar_min_dist.py   `

Computes the minimum radial LiDAR distance for each frame. This serves as a safety-relevant numerical feature for TD3.

### Step 3.3 — Extract Vehicle Metadata

Script: extract\_metadata.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python extract_metadata.py   `

Parses:

*   timestamps
    
*   ego vehicle positions
    
*   yaw
    
*   speed
    
*   yaw\_rate
    

Metadata is stored in metadata\_correct.pkl and used later to approximate acceleration and steering characteristics.

### Step 3.4 — Generate Image and Multi-Modal Embeddings

Script: generate\_camera\_lidar\_embeddings.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python generate_camera_lidar_embeddings.py   `

This step performs:

1.  ViT-B/16 feature extraction
    
2.  CLIP cosine similarity with weather prompts
    
3.  One-hot weather encoding
    
4.  Concatenation of ViT + lidar + weather features
    

The resulting multi-modal embedding is saved as scene\_embeddings\_with\_weather.npy.

4\. Reinforcement Learning Pipeline (TD3)
-----------------------------------------

### Step 4.1 — Build the TD3 Replay Buffer

Script: build\_td3\_buffer.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python build_td3_buffer.py   `

This script:

*   Loads multi-modal scene embeddings
    
*   Loads LiDAR min distance
    
*   Loads metadata (speed, yaw\_rate)
    
*   Generates synthetic expert actions (build\_synthetic\_expert.py)
    
*   Normalizes embeddings and constructs full state vectors
    
*   Computes rewards using reward\_function.py
    
*   Saves:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   states.npy  next_states.npy  actions.npy  rewards.npy  dones.npy   `

The complete buffer is stored in td3\_buffer\_final/.

### Step 4.2 — Train the TD3 Agent

Script: train\_td3.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python train_td3.py   `

This trains the policy over the replay buffer and produces:

*   td3\_policy.pth
    
*   critic loss curve
    
*   actor loss curve
    
*   reward history
    
*   training plots in PNG and PDF formats
    

### Step 4.3 — Evaluate the Learned Policy

Script: evaluate\_td3.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python evaluate_td3.py   `

Computes:

*   Mean Squared Error (MSE) between predicted and expert actions
    
*   Mean Absolute Error (MAE)
    
*   Example actions over a few frames
    

### Step 4.4 — Visualize Agent Behavior (Toy Environment)

Script: visualize\_agent.py

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python visualize_agent.py   `

This creates a simple toy driving visualization that demonstrates how the learned TD3 policy behaves under varying conditions.A video is saved as:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   td3_buffer_final/demo.mp4   `

5\. State and Action Design
---------------------------

### State Vector

Each state contains:

*   768-dim ViT feature (z-score normalized)
    
*   LiDAR minimum distance
    
*   Ego speed
    
*   Yaw rate
    
*   5-dim one-hot weather vector
    

Total dimensionality: 776

### Action Vector

Two continuous outputs:

*   Acceleration
    
*   Steering rate
    

### Reward

The reward function penalizes:

*   Unsafe speed in fog/rain/night
    
*   Steering in low visibility
    
*   Proximity to obstacles
    

It encourages slower, smoother movement in poor conditions and safe driving in general.

6\. Repository Structure
------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   code/      build_synthetic_expert.py      build_td3_buffer.py      evaluate_td3.py      reward_function.py      td3_agent.py      td3_networks.py      train_td3.py      visualize_agent.py      extract_frames.py      extract_lidar_min_dist.py      extract_metadata.py      generate_camera_lidar_embeddings.py  embeddings/      scene_embeddings_with_weather.npy  lidar_min_dist/  metadata/  models/  results/  td3_buffer_final/   `

7\. Summary
-----------

This project demonstrates that reinforcement learning can leverage multi-modal scene representations to learn weather-sensitive driving behavior. By combining vision embeddings, LiDAR geometry, weather classification, and vehicle metadata, the TD3 agent learns a consistent action pattern that reflects both expert priors and reward shaping.

The project serves as a complete implementation-based study of RL applied to autonomous driving on real-world perception data, following the principles of sequential decision-making learned in the course.