# GitHub Upload Steps

## Option 1: Upload using GitHub website

1. Create a new GitHub repository.
2. Recommended repository name:

```text
european-bank-churn-analytics
```

3. Keep the repository public if you want to deploy on Streamlit Cloud easily.
4. Click **Add file** > **Upload files**.
5. Drag and drop all files from this project folder.
6. Click **Commit changes**.

## Option 2: Upload using Git commands

```bash
git init
git add .
git commit -m "Initial commit - European Bank Churn Streamlit App"
git branch -M main
git remote add origin https://github.com/your-username/european-bank-churn-analytics.git
git push -u origin main
```

## Deploy on Streamlit Cloud

1. Go to Streamlit Cloud.
2. Click **New app**.
3. Select your GitHub repository.
4. Set main file path as:

```text
app.py
```

5. Click **Deploy**.
