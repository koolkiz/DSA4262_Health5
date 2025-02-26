# DSA4262_Health5



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