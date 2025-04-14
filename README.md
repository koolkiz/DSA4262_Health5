# DSA4262_Health5

<<<<<<< HEAD


## Setting Up Virtual Environment

To set up your development environment, follow these steps:

1. **Create a Python Virtual Environment**
   ```sh
   python -m venv venv
   ```

2. **Activate the Virtual Environment**
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```


## Developement  Guidelines

1. **Never Work Directly on the `main` Branch**

2. **Always pull the lastest change**

3. **Create a New Branch or Use an Existing One**
    ```sh
    git checkout -b feature-branch
    ```

    OR switch to your existing branch:

    ```sh
    git checkout existing-branch
    ```

4. **Add New Dependencies**

    If you install new packages, update `requirements.txt`:
    ```sh
    pip freeze > requirements.txt
    ```

5. **Submit a Pull Request (PR)**
- Commit and Push your branch:
    ```sh
    git add
    git commit --m "add ur commit msg"
    git push
    ```
- Open a Pull Request on GitHub.
- Request for code reviewers before merging.



Happy coding! 🚀
=======
This branch contains the experimental code for modelling, specifically for NLP and Vision Modellings for feature extraction from Clinical Notes and Spectral Mammography Images respectively, including the Attention-Based Fusion Layer Mechanism.

## Project Structure

```
experiment/
│
├── main.py                               # Main script to run the full workflow
├── requirements.txt                      # List of Python packages for dependencies
├── README.md                             # Project documentation
│
├── data/
│   └── notes/                            # Clinical Notes data
│
├── model/
│   └── nlp.py                            # NLP Feature Extraction Module - Clinical Notes
│
├── processing/
│   └── nlp_preprocessor.py               # Load and Preprocess Clinical Notes
```

## Progress Log
| Date       | Progress              | Remarks                              |
|:-----------|:----------------------|:-------------------------------------|
| 24-02-2025 | Initialise NLP module | Started with ClinicalBERT, yet to check performance |

>>>>>>> 14a197524a22a8136b58e53999adfd0eed762ce1
