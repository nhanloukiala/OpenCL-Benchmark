#ifndef BITMAP_H
#define BITMAP_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#pragma pack(push,1)

// *.bmp have a file type of '0x424d'
static const short bitMapID = 19778;

#define true 1
#define false 0
#define SUCCESS 1
#define FAILURE -1

/**
 * uchar4
 * struct implements a vector of chars
 * on CPU architectures, they could be aligned to 4 - 8 bytes
 */
struct uchar4 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
} ;
typedef struct uchar4 uchar4;
/**
 * color palette of type uchar4
 */
typedef uchar4 ColorPalette;

/**
 * Based on the windows' bmp header info,
 * i create the bitmap header structure 
 */
struct BitMapHeader {
    short id;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} ;

typedef struct BitMapHeader BitMapHeader;

/**
 * Based on the windows' bmp header info,
 * i create the bitmap body structure
 */
struct BitMapInfoHeader {
    int sizeInfo;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} ;

typedef struct BitMapInfoHeader BitMapInfoHeader;

/**
 *class Bitmap used to load a bitmap image from a file.
 */
struct BitMap {
    BitMapHeader header;
    BitMapInfoHeader infoHeader;
    uchar4 * pixels_;				/** Pixel Data */
    int numColors_;					/** Number of colors */
    ColorPalette * colors_;			/** Color Data */
    int isLoaded_;					/** If Bitmap loaded */
} ;

typedef struct BitMap BitMap;

void cleanUp(BitMap* bmp) {
    if (bmp->pixels_ != NULL) free(bmp->pixels_);
    if (bmp->colors_ != NULL) free(bmp->colors_);
    bmp->pixels_ = NULL;
    bmp->colors_ = NULL;
    bmp->isLoaded_ = false;
}

int colorIndex(uchar4 color, BitMap* bmp) {
    for (int i = 0; i < bmp->numColors_; i++) {
        if (bmp->colors_[i].x == color.x &&
            bmp->colors_[i].y == color.y &&
            bmp->colors_[i].z == color.z &&
            bmp->colors_[i].w == color.w) {
            return i;
        }
    }
    return SUCCESS;
}
void load(const char * filename, BitMap* bmp) {
    // Release any existing resources
    //cleanUp(bmp);

    // Open BMP file
    FILE * fd = fopen(filename, "rb");

    // Opened OK
    if (fd != NULL) {
        // Read header
        fread((BitMapHeader *)&(bmp->header), sizeof(BitMapHeader), 1, fd);

        // Failed to read header
        if (ferror(fd)) {
            fclose(fd);
            return;
        }

        // Confirm that we have a bitmap file
        if (bmp->header.id != bitMapID) {
            fclose(fd);
            return;
        }

        // Read map info header
        fread((BitMapInfoHeader *)&(bmp->infoHeader), sizeof(BitMapInfoHeader), 1, fd);

        // Failed to read map info header
        if (ferror(fd)) {
            fclose(fd);
            return;
        }

        // No support for compressed images
        if (bmp->infoHeader.compression) {
            fclose(fd);
            return;
        }

        // Support only 8 or 24 bits images
        if (bmp->infoHeader.bitsPerPixel < 8) {
            fclose(fd);
            return;
        }
	
        // Store number of colors
	        bmp->numColors_ = 1 << bmp->infoHeader.bitsPerPixel;
	
	        //load the palate for 8 bits per pixel
	        if(bmp->infoHeader.bitsPerPixel == 8) {
	            bmp->colors_ = (ColorPalette*) malloc(sizeof(ColorPalette) * bmp->numColors_);
	            if (bmp->colors_ == NULL) {
	                fclose(fd);
	                return;
	            }
	            fread( (char *)bmp->colors_, bmp->numColors_ * sizeof(ColorPalette), 1, fd);
	
	            // Failed to read colors
	            if (ferror(fd)) {
	                fclose(fd);
	                return;
	            }
	        }
	        // Allocate buffer to hold all pixels
	        unsigned int sizeBuffer = bmp->header.size - bmp->header.offset;
	        unsigned char * tmpPixels = (unsigned char*)malloc(sizeof(unsigned char) * sizeBuffer);
	
	        if (tmpPixels == NULL) {
	            free(bmp->colors_);
	            bmp->colors_ = NULL;
	            fclose(fd);
	            return;
	        }
	
	        // Read pixels from file, including any padding
	        fread(tmpPixels, sizeBuffer * sizeof(unsigned char), 1, fd);
	
	        // Failed to read pixel data
	        if (ferror(fd)) {
	            free(bmp->colors_);
	            bmp->colors_ = NULL;
	            free(tmpPixels);
	            fclose(fd);
	            return;
	        }
	
	        // Allocate image
	        bmp->pixels_ = (uchar4*) malloc(sizeof(uchar4) * bmp->infoHeader.width * bmp->infoHeader.height);
	        if (bmp->pixels_ == NULL) {
	            free(bmp->colors_);
	            bmp->colors_ = NULL;
	            free(tmpPixels);
	            fclose(fd);
	            return;
	        }
	        // Set image, including w component (white)
	        memset(bmp->pixels_, 0xff, bmp->infoHeader.width * bmp->infoHeader.height * sizeof(uchar4));
	
	        unsigned int index = 0;
	        for(int y = 0; y < bmp->infoHeader.height; y++) {
	            for(int x = 0; x < bmp->infoHeader.width; x++) {
	                // Read RGB values
	                if (bmp->infoHeader.bitsPerPixel == 8) {
	                    bmp->pixels_[(y * bmp->infoHeader.width + x)] = bmp->colors_[tmpPixels[index++]];
	                }
	                else { // 24 bit
	                    bmp->pixels_[(y * bmp->infoHeader.width + x)].z = tmpPixels[index++];
	                    bmp->pixels_[(y * bmp->infoHeader.width + x)].y = tmpPixels[index++];
	                    bmp->pixels_[(y * bmp->infoHeader.width + x)].x = tmpPixels[index++];
	                }
	            }
	
	            // Handle padding
	            for(int x = 0; x < (4 - (3 * bmp->infoHeader.width) % 4) % 4; x++) {
	                index++;
	            }
	        }
	
	        // Loaded file so we can close the file.
	        fclose(fd);
	        //free(tmpPixels);
	
	        // Loaded file so record this fact
	        bmp->isLoaded_  = true;
	    }
    }

    int writeA(const char * filename, BitMap* bmp) {
	    if (!bmp->isLoaded_) {
	        return false;
	    }
	
	    // Open BMP file
	    FILE * fd = fopen(filename, "wb");
	
	
	    // Opened OK
	    if (fd != NULL) {
	        // Write header
	        fwrite((BitMapHeader *)&(bmp->header), sizeof(BitMapHeader), 1, fd);
	
	        // Failed to write header
	        if (ferror(fd)) {
	            fclose(fd);
	            return false;
	        }
	
	        // Write map info header
	        fwrite((BitMapInfoHeader *)&(bmp->infoHeader), sizeof(BitMapInfoHeader), 1, fd);
	
	        // Failed to write map info header
	        if (ferror(fd)) {
	            fclose(fd);
	            return false;
	        }
	
	        // Write palate for 8 bits per pixel
	        if(bmp->infoHeader.bitsPerPixel == 8) {
	            fwrite( (char *)bmp->colors_, bmp->numColors_ * sizeof(ColorPalette), 1, fd);
	
	            // Failed to write colors
	            if (ferror(fd)) {
	                fclose(fd);
	                return false;
	            }
	        }
	
	        for(int y = 0; y < bmp->infoHeader.height; y++) {
	            for(int x = 0; x < bmp->infoHeader.width; x++) {
	                // Read RGB values
	                if (bmp->infoHeader.bitsPerPixel == 8) {
	                    fputc( colorIndex( bmp->pixels_[(y * bmp->infoHeader.width + x)],bmp), fd);
	                }
	                else { // 24 bit
	                    fputc(bmp->pixels_[(y * bmp->infoHeader.width + x)].z, fd);
	                    fputc(bmp->pixels_[(y * bmp->infoHeader.width + x)].y, fd);
	                    fputc(bmp->pixels_[(y * bmp->infoHeader.width + x)].x, fd);
	
	                    if (ferror(fd)) {
	                        fclose(fd);
	                        return false;
	                    }
	                }
	            }
	
	            // Add padding
	            for(int x = 0; x < (4 - (3 * bmp->infoHeader.width) % 4) % 4; x++) {
	                fputc(0, fd);
	            }
	        }
	
	        return true;
	    }
	    return false;
    }

    int writeB(const char * filename, int width, int height, unsigned int *ptr) {
	    FILE * fd = fopen(filename, "wb");
	
	    int alignSize  = width * 4;
	    alignSize ^= 0x03;
	    alignSize ++;
	    alignSize &= 0x03;
	
	    int rowLength = width * 4 + alignSize;
	
	    // Opened OK
	    if (fd != NULL) 
	    {
	        BitMapHeader *bitMapHeader = (BitMapHeader*)malloc(sizeof(BitMapHeader));
	        bitMapHeader->id = bitMapID;
	        bitMapHeader->offset = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader);
	        bitMapHeader->reserved1 = 0x0000;
	        bitMapHeader->reserved2 = 0x0000;
	        bitMapHeader->size = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader) + rowLength * height;
	        // Write header
	        fwrite(bitMapHeader, sizeof(BitMapHeader), 1, fd);
	        // Failed to write header
	        if (ferror(fd)) 
	        {
	            fclose(fd);
	            return false;
	        }
	
	        BitMapInfoHeader *bitMapInfoHeader = (BitMapInfoHeader*)malloc(sizeof(BitMapInfoHeader));
	        bitMapInfoHeader->bitsPerPixel = 32;
	        bitMapInfoHeader->clrImportant = 0;
	        bitMapInfoHeader->clrUsed = 0;
	        bitMapInfoHeader->compression = 0;
	        bitMapInfoHeader->height = height;
	        bitMapInfoHeader->imageSize = rowLength * height;
	        bitMapInfoHeader->planes = 1;
	        bitMapInfoHeader->sizeInfo = sizeof(BitMapInfoHeader);
	        bitMapInfoHeader->width = width; 
	        bitMapInfoHeader->xPelsPerMeter = 0;
	        bitMapInfoHeader->yPelsPerMeter = 0;
	
	        // Write map info header
	        fwrite(bitMapInfoHeader, sizeof(BitMapInfoHeader), 1, fd);
	
	        // Failed to write map info header
	        if (ferror(fd)) 
	        {
	            fclose(fd);
	            return false;
	        }    
	        unsigned char buffer[4];
	        int x, y;
	
	        for (y = 0; y < height; y++)
	        {
	            for (x = 0; x < width; x++, ptr++)
	            {
	                if( 4 != fwrite(ptr, 1, 4, fd)) 
	                {
	                    fclose(fd);
	                    return false;
	                }
	            }
	            memset( buffer, 0x00, 4 );
	
	            fwrite( buffer, 1, alignSize, fd );
	        }
	
	        fclose( fd );
	        return true;
	    }
	
	    return false;
    }

    int getWidth(BitMap* bmp) {
        if (bmp->isLoaded_) {
            return bmp->infoHeader.width;
        }
        else {
            return -1;
        }
    }

    
	int getNumChannels(BitMap* bmp) {
	    if (bmp->isLoaded_) 
	    {
	        return bmp->infoHeader.bitsPerPixel / 8;
	    }
	    else 
	    {
	        return FAILURE;
	    }
    }

    int getHeight(BitMap* bmp){
        if (bmp->isLoaded_) {
            return bmp->infoHeader.height;
        }
        else {
            return -1;
        }
    }

    uchar4 * getPixels(BitMap *bmp) { return bmp->pixels_; }

    int isLoaded(BitMap *bmp) { return bmp->isLoaded_; }

#pragma pack(pop)
#endif

