# NEAT XOR Visualization

This project implements a minimal, extensible NEAT (NeuroEvolution of Augmenting Topologies) algorithm in Python (NumPy) for solving the XOR problem. It features modular code, robust logging, and visualization of the evolving neural network and fitness progress.

## Features

- NEAT algorithm for XOR
- Cycle prevention and robust crossover
- Visualization of the evolving network and fitness curve
- GIF and video (MP4) export of the evolution process
- All metadata for visualization is saved to `output/progress.json`

## Usage

1. **Run NEAT and log progress:**

   ```sh
   python neat_xor.py
   ```

   This will generate `output/progress.json` with all metadata for visualization.

2. **Create a GIF or video of the evolution:**

   ```sh
   python make_gif.py --metadata_path output/progress.json --gif_path output/progress.gif --duration_ms 60000
   # Or to save as video:
   python make_gif.py --metadata_path output/progress.json --gif_path output/progress.mp4 --duration_ms 60000
   ```

3. **View the result:**
   - The GIF will be saved as [`output/progress.gif`](output/progress.gif).
   - The video (if chosen) will be saved as `output/progress.mp4`.

## Example Output

![NEAT XOR Evolution GIF](output/progress.gif)

## Requirements

- Python 3.7+
- numpy
- networkx
- matplotlib
- imageio>=2.9

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Notes

- If the GIF plays too fast in your viewer, try using a video format (MP4) or use a tool like `gifsicle` to adjust the frame delay.
- The code is modular and can be extended for other NEAT experiments.

---

Created by [Your Name]
