import os

# check if "applications.yml" file exists and if so, remove it
if os.path.exists("slides.yml"):
    os.remove("slides.yml")
