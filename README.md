This repository contains the code used for the experiments in the submitted paper.

The implementation is based on the public PINNacle benchmark:
https://github.com/i207M/PINNacle

To reproduce the experiments:

1. Install dependencies:

   pip install -r requirements.txt

2. Create a .env file in the repository root for Comet logging:

    COMET_API_KEY=your_api_key
    COMET_WORKSPACE=your_workspace

3. Run an experiment:

    bash ./experiments/Wave/wave1d_parallel.sh
