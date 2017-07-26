import numpy as np
from PIL import Image
from keras.models import Model, load_model
from keras.preprocessing import sequence
from utils import *
import os
import copy


# visualiation for
exp_id = "00054"
protein_id = 'P27361'
layers_to_predict = ['input_1', 'embedding_1', 'convolution1d_1', 'activation_1']
window_size = 30


# support stuff
results_dir = "../results"
sequences_file = "../data/protein-seqs-2017-01-26-191058.txt"
vis_directory = results_dir + "/network_vis/"





class NetworkVisualization():
    def __init__(self):
        pass

    def create_input_image(self, layers):
        layer_separation = 20 # in pixels
        small_layer_resize_factor = 20
        small_layer_pixel_threshold = 10

        layer_images = []
        for layer in layers:
            layer_img = Image.fromarray(np.uint8(layer * 255))

            # print layer.shape
            if layer.shape[0] <= small_layer_pixel_threshold:
                layer_img = layer_img.resize(
                                    (   layer.shape[1] * small_layer_resize_factor,
                                        layer.shape[0] * small_layer_resize_factor  ),
                                    Image.NEAREST)
            layer_images.append(layer_img)

        widths, heights = zip(*(i.size for i in layer_images))
        total_width = sum(widths) + (layer_separation * len(layers))
        max_height = max(heights)

        new_im = Image.new('L', (total_width, max_height))
        x_offset = 0
        for im in layer_images:
          new_im.paste(im, (x_offset, ((max_height - im.size[1])/2)))
          x_offset += im.size[0] + layer_separation

        return new_im


    def create_input_layers(self, layer_outputs, input_index, layers_to_predict, model, scale_factor=1, clip_first=None, skip_clipping_num=1):
        vis_layers = [ l[input_index] for l in layer_outputs ]

        output_layer_index = len(layers_to_predict) - 1
        max_sequence_size = model.input_shape[1]

        # reshape input and output layers since None sizes aren't allowed by PIL
        vis_layers[0] = vis_layers[0].reshape(max_sequence_size, 1)
        output_size = model.output_shape[1]
        vis_layers[output_layer_index] = vis_layers[output_layer_index].reshape(output_size,1)

        # normalize input between (0,1)
        vis_layers[0] = vis_layers[0] / 23

        if clip_first != None:
            for i in range(len(vis_layers) - skip_clipping_num):
                # not clipping the last few layers
                vis_layers[i] = vis_layers[i][clip_first:]

        new_im = self.create_input_image(vis_layers)
        return new_im.resize((new_im.size[0] * scale_factor, new_im.size[1] * scale_factor), Image.NEAREST)


    def get_layers_output(self, model, layer_names, data):
        layer_outputs = []
        num_points = len(data)
        max_sequence_size = model.input_shape[1]

        data = np.array(data)
        data = data.reshape(num_points, max_sequence_size)
        print data.shape

        # predict each layer's output and return combined result
        for layer_name in layer_names:
            intermediate_layer_model = Model(input=model.input,
                                             output=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict(data)
            layer_outputs.append(intermediate_output)

        return layer_outputs

    def create_sliding_windows(self, max_sequence_size, original_size, window_size, X):
        win_start = max_sequence_size - original_size
        all_X = []

        for i in range(max_sequence_size - win_start - (window_size-1)):  # stop when last window starts getting cut
            win_start_cur = i + win_start
            win_end_cur = win_start_cur + window_size
            this_copy = np.copy(X)
            # this_copy = copy.deepcopy(X)
            # print win_start_cur, win_end_cur

            # zero out everything before and after window
            this_copy[0:win_start_cur] = 0
            this_copy[win_end_cur:] = 0
            # print this_copy.nonzero()

            all_X.append(this_copy)
        return all_X

if __name__ == '__main__':
    model_filename = results_dir + "/" + exp_id +"-saved-model.h5"
    model = load_model(model_filename)


    max_sequence_size = model.input_shape[1]
    X, original_size = get_protein_sequence(sequences_file, protein_id, max_sequence_size)
    X = X[0] # this is stinky ... # FIXME: fix this

    nv = NetworkVisualization()
    X = nv.create_sliding_windows(max_sequence_size, original_size, window_size, X) # make a sequence of sliding windows

    clip_first = max_sequence_size - int(round(original_size / 100.0) * 100.0)
    print "Clip first: ", clip_first

    layer_outputs = nv.get_layers_output(model, layers_to_predict, X)

    scale_factor = 2
    input_index = 0 # instead loop over all the inputs

    # create directory if doesn't exit
    vis_directory_for_exp = vis_directory + exp_id + "-" + protein_id + "/"
    if not os.path.exists(vis_directory_for_exp):
        os.makedirs(vis_directory_for_exp)

    for i in range(len(X)):
        print "Creating image for window: %d / %d" %(i, len(X))
        img = nv.create_input_layers(layer_outputs, i, layers_to_predict, model, scale_factor, clip_first=clip_first, skip_clipping_num=1)
        img.save(vis_directory_for_exp + ("vis-%05d" % i) +".png")
        # if i > 5: break


    # TODO: convert to gif using ffmpeg -i vis-%05d.png output.gif o gif
