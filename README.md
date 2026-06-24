# LEAF-LLM: Lightweight Energy-Aware Framework for LLMS #
The aim of this project is to develop an evaluation framework to understand the energy performance of lightweight LLMs under different inference configurations and hardware platforms, in order to identify factors that most strongly influence energy-per-token and carbon-per-query. 

How can we build a reproducible, defensible, and comparable evaluation framework that isolates and measures the technical factors driving energy consumption and carbon impact during LLM inference across different models, hardware systems, and runtime configurations? 

Research objectives: 

- to determine which variables most influence inference energy efficiency 
- to understand how measurement methodology affects conclusions 
- to establish a comparison criterion between lightweight LLMs 
- to identify whether optimization gains come primarily from: 
  - model architecture, 
  - quantization, 
  - batching, 
  - hardware accelerators, 
  - runtime frameworks, 
  - workload characteristics, 
  - deployment conditions 
