using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace NeuralNetwork
{
    class Process
    {
        //Turns an image to grayscale form and retrun the values of grayscale pixels in a signal form array mapped to [0, 1].
        public static float[] GraySignal(Bitmap image)
        {
            float[] result = new float[image.Height * image.Width];

            int k = 0;
            for (int j = 0; j < image.Height; j++)
            {
                for (int i = 0; i < image.Width; i++)
                {
                    Color pC = image.GetPixel(i, j);
                    float r = pC.R;
                    float g = pC.G;
                    float b = pC.B;
                    result[k] = 1/(r + g + b + 1.1f) ;
                    
                    k++;
                }
            }
            return result;
        }

        //Resizing input image.
        internal static Bitmap ResizeImage(string rootPath, int size)
        {
            try
            {
                Image image = Image.FromFile(rootPath);

                var destRect = new Rectangle(0, 0, size, size); //creating a rectangle object.
                var destImage = new Bitmap(size, size);

                destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution); //sets the resolution of the new created image.
                                                                                               //resolution, in dots per inch, of the Bitmap.

                using (Graphics graphics = Graphics.FromImage(destImage)) //creating graphics object for alteration.
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                    using var wrapMode = new ImageAttributes();
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
                return destImage;
            }
            catch (DirectoryNotFoundException dirEx)
            {

                throw dirEx;
            }
        }
    }
}
