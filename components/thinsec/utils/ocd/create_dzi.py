#!/usr/bin/env python
# -*- coding: utf-8 -*-

import deepzoom

print 'init..'


# Specify your source image
SOURCE = "/home/oarenas/Desktop/paris-26-gigapixels-small.jpg"
TARGET = "/var/www/html/gmas/assets/data/3/paris.dzi"

# Create Deep Zoom Image creator with weird parameters
creator = deepzoom.ImageCreator(tile_size=128, tile_overlap=2, tile_format="jpg",
                                image_quality=0.8, resize_filter="bicubic")


# Create Deep Zoom image pyramid from source
creator.create(SOURCE, TARGET)

