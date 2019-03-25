import tensorflow as tf
from helper import parse_args, training_dataset, training_model, assign_to_device, average_gradients, do_training


def create_parallel_optimization(model_fn, input_fn, optimizer,
                                 devices=['/gpu:0', '/gpu:1'],
                                 controller="/cpu:0"):
    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                # Compute loss and gradients, but don't apply them yet
                loss = model_fn(input_fn)

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)

            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.

        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss


def main(args):
    dataset = training_dataset(epochs=2)
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    update_op, loss = create_parallel_optimization(training_model,
                                                   input_fn,
                                                   optimizer)

    do_training(update_op, loss)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    training_dataset()