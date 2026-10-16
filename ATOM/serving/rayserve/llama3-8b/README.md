# Llama3-8B with Continuous Batching 
A supplementary guide for the tutorial on serving LLMs with Ray Serve

# Prerequisites
Before we start, RBLN SDK must be installed. For the latest version installation, please check the [docs.rbln.ai](https://docs.rbln.ai/latest/supports/release_note.html).
```
$ pip3 install --extra-index-url https://pypi.rbln.ai/simple rebel-compiler==<VERSION> optimum-rbln==<VERSION> vllm-rbln==<VERSION>
```

# Files
- `material/`
  - Contains the source files extracted from the document page and used in this tutorial.
- `check_env.sh`
  - A Bash script to verify all necessary preparations for running this tutorial properly.
- `setup.sh`
  - A setup script for preparing the required files and configurations for this tutorial.

# Result
You can check the `output` directory, which contains all the necessary preparations for running the tutorial.

# References
[Llama3-8B](https://docs.rbln.ai/software/model_serving/rayserve/tutorial/llama3-8B.html)
