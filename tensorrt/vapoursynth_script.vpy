
import vapoursynth as vs
core = vs.core

# load clip
source_path = "path/to/video.mkv"
clip = core.bs.VideoSource(source=source_path, cachemode=3)

# pad clip to next multiple of 64
w = clip.width
h = clip.height
pad_r = (((w + 63) // 64) * 64) - w
pad_b = (((h + 63) // 64) * 64) - h
clip = core.std.AddBorders(clip, right=pad_r, bottom=pad_b)

# convert to RGBS for vsmlrt. Use RGBH if running an FP16 model
clip = core.resize.Bilinear(clip, format=vs.RGBS, matrix_in_s="470bg")

# inference with 5 input frames
m2 = clip[:2] + clip[:-2]   # shift -2
m1 = clip[:1] + clip[:-1]   # shift -1
p1 = clip[1:] + clip[-1:]   # shift +1
p2 = clip[2:] + clip[-2:]   # shift +2
clip = core.trt.Model([m2, m1, clip, p1, p2], engine_path="path/to/tscunet.engine", num_streams=1)

# crop padding
clip = core.std.Crop(clip, right=pad_r*2, bottom=pad_b*2)

# convert to output format
clip = core.resize.Bilinear(clip, format=vs.YUV420P8, matrix_s="709")

# set output
clip.set_output()

















