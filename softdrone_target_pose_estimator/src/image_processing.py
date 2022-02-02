import numpy as np
from PIL import Image

# These parameters are from the detectron model. They will need
# to modified for different data sets.
MIN_SIZE_TEST = 720
MAX_SIZE_TEST = 1280

def preprocess_input(image):
    h, w = image.shape[:2]
    newh, neww = get_output_shape(h, w, MIN_SIZE_TEST, MAX_SIZE_TEST)
    image = apply_image(newh, neww, image)
    #image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return image

def postprocess_output(input_shape, output_shape, kpts):
    scale_x = output_shape[0] / float(input_shape[0])
    scale_y = output_shape[1] / float(input_shape[1])
    scaled_kpts = kpts.copy()
    scaled_kpts[:,0] = scaled_kpts[:,0]*scale_x
    scaled_kpts[:,1] = scaled_kpts[:,1]*scale_y
    return scaled_kpts.astype(int)

def get_output_shape(oldh, oldw, short_edge_length, max_size):
    """
    Compute the output size given input size and target short edge length.
    """
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_image(new_h, new_w, img, interp=None):
    h, w = img.shape[:2]
    assert len(img.shape) <= 4
    interp_method = interp if interp is not None else Image.BILINEAR

    if img.dtype == np.uint8:
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.resize((new_w, new_h), interp_method)
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
    else:
        # PIL only supports uint8
        if any(x < 0 for x in img.strides):
            img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        shape = list(img.shape)
        shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
        img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
        _PIL_RESIZE_TO_INTERPOLATE_MODE = {
            Image.NEAREST: "nearest",
            Image.BILINEAR: "bilinear",
            Image.BICUBIC: "bicubic",
        }
        mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
        align_corners = None if mode == "nearest" else False
        img = F.interpolate(
            img, (self.new_h, self.new_w), mode=mode, align_corners=align_corners
        )
        shape[:2] = (new_h, new_w)
        ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

    return ret

if __name__=="__main":
    # TODO: Don't hardcode image path.
    img_cv = cv2.imread("/home/subella/test/medkit_image_125.png")
    img_preproc = preprocess_image(img_cv)


