# Vision ID Photo Tool (\`vision.py\`) Quick Manual (macOS)

This tool processes a folder of photos to produce **TASS-ready ID photos** automatically.

## 1. Installation (One-time Setup)

### Step 1: Create the `bin` directory

Open **Terminal** and run the following command to ensure your local bin folder exists:

```bash
mkdir -p ~/bin
```

### Step 2: Unzip the bundle

Assuming the zip file is in your **Downloads** folder, run:

```bash
unzip -o ~/Downloads/vision-bundle.zip -d ~/bin
```

### Step 3: Make the tool executable

Give the system permission to run the script:

```bash
chmod +x ~/bin/vision.py
```

### Step 4: Confirm installation

Verify the file is in the correct location:

```bash
ls -l ~/bin/vision.py
```

---

## 2. Directory Structure

For the tool to function, the following three files **must** remain together in `~/bin`:

| File                                     | Description                           |
|------------------------------------------|---------------------------------------|
| `vision.py`                                | The main execution script             |
| `deploy.prototxt`                          | Configuration file for face detection |
| `res10_300x300_ssd_iter_140000.caffemodel` | The trained AI model                  |

---

## 3. Usage Instructions

1. **Prepare Input:** Put your raw photos into a folder (e.g., `~/Downloads/photos_in`).
2. **Prepare Output:** Create an empty folder for the results (e.g., `~/Downloads/photos_out`).
3. **Run the Tool:** Execute the script by passing the input and output paths:

```bash
~/bin/vision.py ~/Downloads/photos_in ~/Downloads/photos_out
```

*The process is complete when the terminal prompt returns.*

---

## 4. Confirming Results

Open your output folder (`~/Downloads/photos_out`) and verify:

* [ ] All processed files are in `.jpg` format.
* [ ] Photos are rotated **upright**.
* [ ] Faces are **centered and framed** correctly for ID requirements.

---

## 5. Troubleshooting

* **"No such file or directory":** * Double-check that your input and output folder paths are typed correctly.
* Ensure the output folder was created *before* running the command.
* **Output folder is empty:** * Ensure the input folder contains images directly.
* The tool supports `.jpg`, `.jpeg`, and `.png` but **does not** scan subfolders.