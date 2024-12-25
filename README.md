
# Image Tile Mosaic

This allows you to recreate a target image with blocks from one or more source images. We take the Discrete Cosine Transform (DCT) of each block in the target image, and compare that to each DCT block in the source images using Mean Squared Error (MSE). We then tile the target image with blocks from the source image that had the lowest MSE (without replacement). It outputs the reconstructed image made up of blocks from the source image, and annotated versions of the source images, describing where each block should be placed to form the reconstructed image. 

This works because the DCT transforms an image into its frequency components, and two images with similar frequency components contain similar shapes. So by replacing the target image blocks with blocks that contain similar shapes, we retain the objects in the target image. 

## Usage

Set the fields in the top of image_tile_mosaic.py to change the block size or resize the images. Then run the script. The first argument should be the target image and the next arguments should be any number of source images. 

```
python image_tile_mosaic.py target.jpg source1.jpg source2.jpg source3.jpg
```

## Results

<div float="left">
    <img src="./assets/target.jpg" alt="target" height="200">
    <img src="./assets/source.jpg" alt="source" height="200">
    <img src="./assets/reconstruction.png" alt="reconstruction" height="200">
    <img src="./assets/source_annotated_closeup.png" alt="source_annotated" height="200">
</div>
<br>

Left to right:
target image, source image, reconstruction, annotated source image closeup

The annotated source image outlines the used blocks in white and contains three bits of information, row, column, and rotation. For instance, "6 8 L" means this block should be placed on the reconstructed image at row 6 (0-indexed) column 8 (0-indexed) and rotated 90 degrees to the Left (counterclockwise). 

