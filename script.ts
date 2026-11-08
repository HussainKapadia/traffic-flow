import fs from "node:fs";
import path from "node:path";

const ROOT = import.meta.dirname;
const SPLIT_RATIO = 0.15;
const DRY_RUN = process.argv.includes("--dry-run");

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".bmp", ".webp"]);

const trainImages = path.join(ROOT, "train", "images");
const trainLabels = path.join(ROOT, "train", "labels");
const testImages = path.join(ROOT, "test", "images");
const testLabels = path.join(ROOT, "test", "labels");

fs.mkdirSync(testImages, { recursive: true });
fs.mkdirSync(testLabels, { recursive: true });

// Only consider images that have a matching label file
const pairs = fs
  .readdirSync(trainImages)
  .filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase()))
  .map((image) => ({ image, label: path.parse(image).name + ".txt" }))
  .filter(({ image, label }) => {
    const exists = fs.existsSync(path.join(trainLabels, label));
    if (!exists) console.warn(`Skipping ${image}: no matching label`);
    return exists;
  });

// Fisher-Yates shuffle
for (let i = pairs.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
}

const count = Math.round(pairs.length * SPLIT_RATIO);
const selected = pairs.slice(0, count);

for (const { image, label } of selected) {
  if (DRY_RUN) {
    console.log(`would move ${image} + ${label}`);
    continue;
  }
  fs.renameSync(path.join(trainImages, image), path.join(testImages, image));
  fs.renameSync(path.join(trainLabels, label), path.join(testLabels, label));
}

console.log(
  `${DRY_RUN ? "[dry run] Would move" : "Moved"} ${count} of ${pairs.length} image/label pairs from train to test.`
);
