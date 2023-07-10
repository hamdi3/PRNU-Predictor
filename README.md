# PRNU Predictor

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hamdi3/PRNU-Predictor/blob/main/LICENSE)
[![Python 3.10.11](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![PyTorch 2.0.1](https://img.shields.io/badge/pytorch-2.0.1-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Streamlit 1.23.1](https://img.shields.io/badge/streamlit-1.23.1-FF4B4B.svg)](https://streamlit.io/)
[![Docker Support](https://img.shields.io/badge/docker-support-2496ED.svg)](https://www.docker.com/)
![Built with Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red.svg)


Welcome to the PRNU Predictor GitHub repository! This project aims to identify the original device used to capture images by leveraging the PRNU (Photo-Response Non-Uniformity) values. The PRNU values serve as unique fingerprints for different imaging devices, making it possible to determine the device that captured a particular image.

## Background

The idea behind this project is inspired by the research conducted by the Politecnico di Milano, available in their repository [prnu-python](https://github.com/polimi-ispl/prnu-python/tree/master/prnu). We built upon their work and expanded the functionality to create a user-friendly web application for predicting the device that captured an image based on its PRNU signature.

## Features

- Predicts the original device used to capture an image by analyzing its PRNU values.
- Supports various imaging devices (Training the model further to support more devices is always welcomed 😉):
  - FrontCamera-GalaxyA13-225951
  - FrontCamera-GalaxyA13-225952
  - Logitech Brio210500
  - Logitech Brio210504
  - Logitech Brio210506
  - Logitech C50596011268
  - Logitech C50596011268_2
  - Logitech C50596011268_3
  - Nikon_Zfc
  - RückCamera-GalaxyA13-225951
  - Rückcamera-GalaxyA13-225952
- Deployed web application accessible at [https://prnu-predictor.streamlit.app/](https://prnu-predictor.streamlit.app/).
- Docker support for easy deployment.

## Getting Started

To get started with the PRNU Predictor, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/prnu-predictor.git
   ```

2. Install the required dependencies. We recommend using a virtual environment:
   ```bash
   cd prnu-predictor
   python3.10 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Launch the web application:
   
   ```bash
   streamlit run app.py
   ```
   
4. Access the web application by opening `http://localhost:8501` in your browser.

## Docker Support

The PRNU Predictor also provides Docker support for easy deployment. To build and run the Docker image, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t prnu-predictor .
   ```
  
2. Run the Docker container:
   
   ```bash
   docker run -p 8501:8501 prnu-predictor
   ```
   
4. Access the web application by opening `http://localhost:8501` in your browser.

## Contributing

Contributions are welcome and greatly appreciated! To contribute to the PRNU Predictor project, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make the desired changes and commit them:
   
   ```bash
   git commit -m "Add my feature"
   ```

4. Push to the branch:
      
   ```bash
   git push origin feature/my-feature
   ```

5. Open a pull request in the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/your-username/prnu-predictor/blob/main/LICENSE) file for more details.

## Acknowledgments

I would like to express my gratitude to the researchers at the Politecnico di Milano for their contributions to the PRNU analysis and their [prnu-python](https://github.com/polimi-ispl/prnu-python) repository, which served as the foundation for this project.

## Contact

If you have any questions, suggestions, or feedback, please feel free to contact me:

- Your Name
  - GitHub: [github.com/hamdi3](https://github.com/hamdi3)

I'm open to collaboration and look forward to hearing from you!

---

Thank you for visiting the PRNU Predictor repository. I hope you find it useful and informative. Happy device identification using PRNU values!

