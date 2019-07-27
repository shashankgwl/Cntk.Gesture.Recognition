from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt
import numpy as np

import cntk.io.transforms as xforms
import cv2 as cv
try: 
    from urllib.request import urlopen 
except ImportError: 
    from urllib import urlopen

import cntk as C

#C.device.try_set_default_device(C.device.gpu(0))
C.device.try_set_default_device(C.device.gpu(0))
image_height = 50
image_width  = 50
num_channels = 1
num_classes  = 3
win1 = "win1"
win2 = "win2"
cap = cv.VideoCapture(0)

train_file= "D:\Shashank\One Drive Business\OneDrive - Microsoft\SHASHANK\learning\ML\CNTK\data\signature\\train_201B.txt"
test_file= "D:\Shashank\One Drive Business\OneDrive - Microsoft\SHASHANK\learning\ML\CNTK\data\signature\\test_201B.txt"

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)
    #print("resized shape is {0}".format(resized.shape))

    # return the resized image
    return resized

def create_basic_model(input, out_dims):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        model = C.layers.Sequential([
            C.layers.For(range(3), lambda i: [
                C.layers.Convolution((5,5), [32,32,64][i], pad=True),
                C.layers.MaxPooling((3,3), strides=(2,2))
                ]),
            C.layers.Dense(64),
            C.layers.Dropout(0.25),
            C.layers.Dense(out_dims, activation=C.softmax)
        ])

        return model(input)




def create_reader(map_file, train):
    print("Reading map file:", map_file)
    
    
    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    # train uses data augmentation (translation only)
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8) 
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')
    ]
    # deserializer
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(
        features = C.io.StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = C.io.StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))
    
reader_train = create_reader(train_file, True)
reader_test  = create_reader(test_file, False)

def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    # Input variables denoting the features and label data
    input_var = C.input_variable((num_channels, image_height, image_width))
    label_var = C.input_variable((num_classes))

    # Normalize the input
    feature_scale = 1./ 255.
    input_var_norm = C.element_times(feature_scale, input_var)
    
    # apply model to input
    z = model_func(input_var_norm, out_dims=3)

    #
    # Training action
    #

    # loss and metric
    ce = C.cross_entropy_with_softmax(z, label_var)
    pe = C.classification_error(z, label_var)

    # training config
    epoch_size     = 50000
    minibatch_size = 20

    # Set training parameters
    lr_per_minibatch       = C.learning_rate_schedule([0.001]*10 + [0.003]*10 + [0.001], 
                                                      C.UnitType.minibatch, epoch_size)
    momentum_time_constant = C.momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.001
    
    # trainer object
    #learner = C.sgd(z.parameters, lr = 0.01, minibatch_size = C.learners.IGNORE)
    learner = C.momentum_sgd(z.parameters, lr = lr_per_minibatch, momentum = momentum_time_constant, l2_regularization_weight=l2_reg_weight)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = C.Trainer(z, (ce, pe), [learner], [progress_printer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    C.logging.log_number_of_parameters(z) ; print()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), 
                                               input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far
            
            # For visualization...            
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
            
            batch_index += 1
        trainer.summarize_training_progress()
        
    #
    # Evaluation action
    #
    epoch_size     = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")
    
    # Visualize training result:
    window_width            = 32
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()
    
    return C.softmax(z)

def eval(pred_op, image_data):
    #print("shape of image is {0}".format(image_data.shape))
    label_lookup = ["One", "two", "three"]
    image_data = np.ascontiguousarray(image_data.reshape(1,50,50),dtype=np.float32)
    
    result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))
    
    # Return top 3 results:
    top_count = 1
    
    result_indices = (-np.array(result)).argsort()[:top_count]
    res = result[result_indices[0]] * 100
    if(res > 40.):
        print("Prediction: {:10s}, confidence: {:.2f}%\n".format(label_lookup[result_indices[0]], result[result_indices[0]] * 100))
    
    
    #print("Top 3 predictions:")
    #for i in range(top_count):
        


pred = train_and_evaluate(reader_train, reader_test, max_epochs=15, model_func=create_basic_model)

cv.namedWindow(win2)
cv.namedWindow("win3")
while 1:    
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv,np.array([0,30,60]),np.array([20,150,255]))
    newmasked = cv.bitwise_and(frame,frame,mask=mask1)
    newmasked= cv.cvtColor(newmasked,cv.COLOR_BGR2GRAY)
    #print(newmasked.shape)
    blurred = cv.GaussianBlur(newmasked,(5,5),0)
    
    canny = cv.Canny(blurred,0,255)
    kernel = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    
    #cv.rectangle(opening,(350,10),(610,228),(255,255,255),1)
    
    
    try:
        ROI = opening[10:228, 350:600]
        image2predict = cv.resize(ROI,(50,50))
        eval(pred,np.array(image2predict,dtype=np.float32))
        cv.imshow(win2, frame)
        cv.imshow("win3",ROI)
        key = cv.waitKey(30)
        if key == ord('q') or key == 27:
            cap.release()
            cv.destroyAllWindows()
            break
    except:
        pass
    
    # draw the book contour (in green)
    #cv.rectangle(ROI,(x,y),(x+w,y+h),(255,255,255),5)
    #ROI = ROI[y:y+h,x:x+w]
    #print(contours.shape)
    #ret, thrsh = cv.threshold(ROI,200,255,cv.THRESH_BINARY)
    #cv.imshow(win1, frame)
    
    

   
    
