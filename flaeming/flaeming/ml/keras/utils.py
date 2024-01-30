def get_model_tag(model):
    model_config = model.get_config()
    model_string = ""

    for layer in model_config["layers"]:
        layer_name = layer["class_name"]
        layer_config = layer["config"]

        if layer_name == "InputLayer":
            continue
        elif layer_name == "Conv2D":
            fltr = layer_config["filters"]
            kernel = layer_config["kernel_size"][0]  # assumes square kernel
            stride = layer_config["strides"][0]  # assumes equal dim strides
            pad = layer_config["padding"]
            model_string += f"{layer_name}_{fltr}-{kernel}-{stride}-{pad}>"
        elif layer_name == "MaxPooling2D":
            kernel = layer_config["pool_size"][0]  # assumes square kernel
            stride = layer_config["strides"][0]  # assumes equal dim strides
            pad = layer_config["padding"]
            model_string += f"{layer_name}-{kernel}-{stride}-{pad}>"
        elif layer_name == "Dense":
            n_units = layer_config["units"]
            activation = layer_config["activation"]
            model_string += f"{layer_name}-{n_units}-{activation}>"
        elif layer_name == "Dropout":
            rate = layer_config["rate"]
            model_string += f"{layer_name}-{rate*100:.0f}>"
        elif layer_name == "Functional":
            model_name = layer_config["name"]
            model_string += f"{model_name}>"
        else:
            model_string += f"{layer_name}>"

    return model_string
