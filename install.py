import os

root = os.path.dirname(__file__)

print("Installing python packages...")
os.system(f"pip3 install -r {os.path.join(root, 'requirements.txt')}")

print("Creating directories...")
os.mkdir(os.path.join(root, "saves"))
os.mkdir(os.path.join(root, "onGAU", "models"))

print("What should be the default model when starting onGAU?")
model = input("Path to model: ")

with open(os.path.join(root, "onGAU", "config.py"), "r+") as f:
    contents = f.read()
    contents = contents.replace('"%default_model%"', repr(model), 1)

    f.seek(0)
    f.truncate()
    f.write(contents)

print("Finished onGAU setup! Happy generating!")
