library(keras)
library(keras3)
library(tensorflow)
install.packages("tfdatasets")

library(tfdatasets)

library(readr)

# Separate labels and image data
labels <- xy_train[[1]]  # Assuming first column is label
x_train <- as.matrix(xy_train[, -1])  # Drop the label column
xy_test <- as.matrix(x_test[,-1])

dim(x_train)
dim(xy_train)

# Normalize pixel values to [0,1] (important for Keras)
x_train <- x_train / 255
xy_test <- xy_test / 255

# Reshape images to 3D array (number of images, height, width, channels)
# Keras in R expects shape: (samples, 28, 28, 1)
x_train <- array_reshape(x_train, dim = c(nrow(x_train), 28, 28, 1))
xy_test <- array_reshape(xy_test, dim = c(nrow(xy_test), 28, 28, 1))

labels_factor <- factor(labels)
labels_int <- as.integer(labels_factor) - 1

y_train <- to_categorical(labels_int)

dim(y_train)
########################## Initial code setup complete

set.seed(42)
n <- nrow(x_train)
train_indicies <- sample(1:n, size = 0.8*n)
dim(x_train)
dim(y_train)

x_train_split <- x_train[train_indicies,,,, drop = FALSE]
y_train_split <- y_train[train_indicies,,drop = FALSE]

x_val_split <- x_train[-train_indicies,,,,drop = FALSE]
y_val_split <- y_train[-train_indicies,,drop = FALSE]


BATCH_SIZE <- 6

# 3. Create TensorFlow Dataset objects
train_dataset <- tensor_slices_dataset(list(x_train_split, y_train_split)) %>%
  dataset_shuffle(buffer_size = 1000) %>%
  dataset_batch(BATCH_SIZE) %>%
  dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

val_dataset <- tensor_slices_dataset(list(x_val_split, y_val_split)) %>%
  dataset_batch(BATCH_SIZE) %>%
  dataset_prefetch(buffer_size = tf$data$AUTOTUNE)

num_classes <- 6

model <- keras_model_sequential() %>%
  
  layer_random_flip("horizontal") %>%
  layer_random_rotation(0.1) %>%
  layer_random_zoom(0.1) %>%
  
  layer_conv_2d(filters = 32, kernel_size = 3, padding = "same", activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)


history <- model %>% fit(
  train_dataset,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = val_dataset,
  validation_steps = 50
)


###################Prediction

xy_test <- array_reshape(xy_test, dim = c(nrow(xy_test), 28, 28, 1))

dim(xy_test)

y_pred <- model %>% predict(xy_test)

dim(y_pred)

save(x_test_final, file = "predictions.RData")
write.csv(x_test_final, file = "predictions.csv", row.names = FALSE)

y_pred
dim(x_test)
y_pred_labels

x_test_final <- x_test
x_test_final[,1] <- y_pred_labels
x_test_final[,1]


# Check shape of predictions
dim(y_pred)  # Equivalent to y_pred.shape

# Get index of maximum probability per prediction
tmp <- apply(y_pred, 1, which.max) - 1  # Subtract 1 for zero-based index like Python

# Define categories
cats <- c("A", "B", "C", "D", "E", "F")

# Map indices to category labels
y_pred_labels <- cats[tmp + 1]  # Add 1 to shift back to R's 1-based indexing

# Create a named list (like a dictionary in Python)
labels_dict <- list(Prediction = y_pred_labels)

# Print results
print(y_pred_labels)
print(length(y_pred_labels))

n <- 10


# Plot grayscale image
image(1:28, 1:28, t(apply(xy_test[n, , , 1], 2, rev)), col = gray.colors(256), axes = FALSE, main = y_pred_labels[n])




################################# Generative data


library(keras)
use_implementation("tensorflow")
library(tensorflow)
library(reticulate)
library(ggplot2)
library(dplyr)
library(readr)

use_condaenv("r-reticulate")

# Set a random seed in R to make it more reproducible 
set.seed(123)

# Set the seed for Keras/TensorFlow
tensorflow::set_random_seed(123)

dim(x_train)

original_dim <- 784L
latent_dim <- 20L
intermediate_dim <- 256L
batch_size<- 6

encoder_inputs <- layer_input(shape = 28 * 28)

x <- encoder_inputs %>%
  layer_dense(intermediate_dim, activation = "relu")

z_mean    <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
                       name = "encoder")


# layer_sampler <- new_layer_class(
#   classname = "Sampler",
#   call = function(z_mean, z_log_var) {
#     epsilon <- tf$random$normal(shape = tf$shape(z_mean))
#     z_mean + exp(0.5 * z_log_var) * epsilon }
# )

layer_sampler <- new_layer_class(
  classname = "Sampler",
  
  initialize = function() {
    super$initialize()
  },
  
  call = function(inputs, mask = NULL) {
    z_mean <- inputs[[1]]
    z_log_var <- inputs[[2]]
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + tf$exp(0.5 * z_log_var) * epsilon
  }
)


latent_inputs <- layer_input(shape = c(latent_dim))

decoder_outputs <- latent_inputs %>%
  layer_dense(intermediate_dim, activation = "relu") %>%
  layer_dense(original_dim, activation = "sigmoid")

decoder <- keras_model(latent_inputs, decoder_outputs,
                       name = "decoder")







model_vae <- new_model_class(
  classname = "VAE",
  
  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <-
      metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <-
      metric_mean(name = "kl_loss")
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),
  
  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
      
      c(z_mean, z_log_var) %<-% self$encoder(data)
      # z <- self$sampler(z_mean, z_log_var)
      z <- self$sampler(list(z_mean, z_log_var))
      
      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_binary_crossentropy(data, reconstruction) %>%
        sum(axis = c(1)) %>%
        mean()
      
      kl_loss <- -0.5 * (1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })
    
    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))
    
    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    
    list(total_loss = self$total_loss_tracker$result(),
         reconstruction_loss = self$reconstruction_loss_tracker$result(),
         kl_loss = self$kl_loss_tracker$result())
  }
)

x_train_flat <- array_reshape(x_train, dim = c(nrow(x_train), 28 * 28))
dim(x_train_flat)

vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(x_train_flat, epochs = 20,
            shuffle = TRUE)

#######################Generate images

# Create a grid of 2D latent points
n <- 15  # number of images per axis
grid_x <- seq(-3, 3, length.out = n)
grid_y <- seq(-3, 3, length.out = n)

# Initialize an empty list to store generated images
generated_images <- list()


for (i in 1:n) {
  for (j in 1:n) {
    z_samples <- matrix(rnorm(latent_dim, 0, 1), ncol = latent_dim)
    decoded_img <- decoder %>% predict(z_samples)
    generated_images[[length(generated_images) + 1]] <- matrix(decoded_img, nrow = 28, ncol = 28)
  }
}



library(grid)
library(gridExtra)
install.packages('gridExtra')

# Convert to grobs
grobs <- lapply(generated_images, function(img) {
  rasterGrob(img, interpolate = TRUE)
})

# Arrange in grid
grid.arrange(grobs = grobs, ncol = n)



dim(xy_train)
total_data <- rbind(x_test_final,xy_train)
dim(total_data)


total_labels <- total_data[[1]]  
total_labels

total_labels_factor <- factor(total_labels)
total_labels_factor

total_labels_int <- as.integer(total_labels_factor) - 1
total_labels_int

dim(class1)

indicies <- which(total_labels_int == 1)
indicies
length(indicies)
class1 <- total_data[indicies,,,, drop = FALSE]

dim(class1[,-1])

total_train_flat <- array_reshape(class1[,-1], dim = c(nrow(class1), 28 * 28))
dim(total_train_flat)

vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(total_train_flat, epochs = 100,
            shuffle = TRUE, batch_size = 6)




















generated_images_list <- list()
dim(y_train)
y_train
labels_int

class_label
x_train[1 == class_label,,,, drop = FALSE]

for (class_label in 0:5) {
  cat("Processing class:", class_label, "\n")
  
  # 1. Filter the training data for the current class
  x_class <- x_train[labels_int == class_label,,,, drop = FALSE]
  
  # 2. Flatten to match VAE input shape (e.g., 784 for 28x28 images)
  x_class_flat <- array_reshape(x_class, c(nrow(x_class), 784))
  
  # 3. Re-initialize a fresh VAE (or reuse if you're resetting weights)
  encoder_inputs <- layer_input(shape = 784)
  x <- encoder_inputs %>%
    layer_dense(256, activation = "relu")
  z_mean <- x %>% layer_dense(2, name = "z_mean")
  z_log_var <- x %>% layer_dense(2, name = "z_log_var")
  encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var))
  
  latent_inputs <- layer_input(shape = 2)
  decoder_outputs <- latent_inputs %>%
    layer_dense(256, activation = "relu") %>%
    layer_dense(784, activation = "sigmoid")
  decoder <- keras_model(latent_inputs, decoder_outputs)
  
  layer_sampler <- new_layer_class(
    classname = "Sampler",
    call = function(z_mean, z_log_var) {
      epsilon <- tf$random$normal(shape = tf$shape(z_mean))
      z_mean + exp(0.5 * z_log_var) * epsilon
    }
  )
  
  model_vae <- new_model_class(
    classname = "VAE",
    initialize = function(encoder, decoder, ...) {
      super$initialize(...)
      self$encoder <- encoder
      self$decoder <- decoder
      self$sampler <- layer_sampler()
      self$total_loss_tracker <- metric_mean(name = "total_loss")
      self$reconstruction_loss_tracker <- metric_mean(name = "reconstruction_loss")
      self$kl_loss_tracker <- metric_mean(name = "kl_loss")
    },
    metrics = mark_active(function() {
      list(self$total_loss_tracker, self$reconstruction_loss_tracker, self$kl_loss_tracker)
    }),
    train_step = function(data) {
      with(tf$GradientTape() %as% tape, {
        c(z_mean, z_log_var) %<-% self$encoder(data)
        z <- self$sampler(z_mean, z_log_var)
        reconstruction <- self$decoder(z)
        reconstruction_loss <- loss_binary_crossentropy(data, reconstruction) %>%
          tf$reduce_sum(axis = 1L) %>%
          tf$reduce_mean()
        kl_loss <- -0.5 * tf$reduce_mean(1 + z_log_var - z_mean^2 - exp(z_log_var))
        total_loss <- reconstruction_loss + kl_loss
      })
      grads <- tape$gradient(total_loss, self$trainable_weights)
      self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))
      self$total_loss_tracker$update_state(total_loss)
      self$reconstruction_loss_tracker$update_state(reconstruction_loss)
      self$kl_loss_tracker$update_state(kl_loss)
      list(total_loss = self$total_loss_tracker$result(),
           reconstruction_loss = self$reconstruction_loss_tracker$result(),
           kl_loss = self$kl_loss_tracker$result())
    }
  )
  
  # 4. Compile and train VAE
  vae <- model_vae(encoder, decoder)
  vae %>% compile(optimizer = optimizer_adam())
  vae %>% fit(x_class_flat, epochs = 15, batch_size = 128, shuffle = TRUE)
  
  # 5. Generate 500 new images from the trained decoder
  z_samples <- matrix(rnorm(500 * 2), ncol = 2)  # latent_dim = 2
  generated_flat <- decoder %>% predict(z_samples)
  generated_images <- array_reshape(generated_flat, c(500, 28, 28))
  
  # 6. Store
  generated_images_list[[as.character(class_label)]] <- generated_images
}
