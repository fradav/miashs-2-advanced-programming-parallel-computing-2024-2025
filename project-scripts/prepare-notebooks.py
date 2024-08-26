import os

# check if "applications.yml" file exists and if so, remove it
if os.path.exists("applications.yml"):
    os.remove("applications.yml")
    
def copy(source, destination):
   with open(source, 'rb') as file:
       myFile = file.read()
   with open(destination, 'wb') as file:
       file.write(myFile)

envfiles = os.getenv("QUARTO_PROJECT_INPUT_FILES")

def solution2app(filepath):
    # split the path
    path_parts = filepath.split(os.sep)
    # check if the path is a solution directory and remove it in the path
    if len(path_parts) > 1 and path_parts[-2] == "Solutions":
        file_name = path_parts[-1]
        if file_name.endswith("-sol.qmd"):
            file_name = file_name.replace("-sol.qmd", ".qmd")
        return os.path.join(*path_parts[:-2], "Applications", file_name)
    else:
        return None
     
for file in envfiles.split("\n"):
    print(file)
    appfile = solution2app(file)
    if appfile:
        if os.path.exists(appfile):
            os.remove(appfile)
        # copy the file to the new name not rename
        copy(file, appfile)
