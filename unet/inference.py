def predict(self, source, patch_size, margin):
        
    def get_coord(shape, size, margin):
        assert len(shape)==len(size)

        shape = np.array(shape)
        size = np.array(size)
        margin = np.array(margin)

        n_tiles = (shape-1)//(size-2*margin)
        if len(n_tiles)==3:
            n_tiles_z, n_tiles_y, n_tiles_x = n_tiles
        elif len(n_tiles)==2:
            n_tiles_z, (n_tiles_y, n_tiles_x) = 0, n_tiles

        for i in range(n_tiles_z+1):
            if len(n_tiles)==3:
                src_start_z = i*(size-2*margin)[0] if i<n_tiles_z else (shape-size)[0]
                src_end_z = src_start_z+size[0]
                left_z = margin[0] if i>0 else 0
                right_z = margin[0] if i<n_tiles_z else 0

            for j in range(n_tiles_y+1):
                src_start_y = j*(size-2*margin)[-2] if j<n_tiles_y else (shape-size)[-2]
                src_end_y = src_start_y+size[1]
                left_y = margin[-2] if j>0 else 0
                right_y = margin[-2] if j<n_tiles_y else 0
                for k in range(n_tiles_x+1):
                    src_start_x = k*(size-2*margin)[-1] if k<n_tiles_x else (shape-size)[-1]
                    src_end_x = src_start_x+size[-1]
                    left_x = margin[-1] if k>0 else 0
                    right_x = margin[-1] if k<n_tiles_x else 0
                    
                    if self.opts.proj_model and len(n_tiles)==3:
                        src_s = (slice(None, None), 
                                 slice(src_start_y, src_end_y), 
                                 slice(src_start_x, src_end_x))
                        trg_s = (slice(src_start_y+left_y, src_end_y-right_y), 
                                 slice(src_start_x+left_x, src_end_x-right_x))
                        mrg_s = (slice(left_y, -right_y if right_y else None), 
                                 slice(left_x, -right_x if right_x else None))

                    elif len(n_tiles)==3:
                        src_s = (slice(src_start_z, src_end_z), 
                                 slice(src_start_y, src_end_y), 
                                 slice(src_start_x, src_end_x))
                        trg_s = (slice(src_start_z+left_z, src_end_z-right_z), 
                                 slice(src_start_y+left_y, src_end_y-right_y), 
                                 slice(src_start_x+left_x, src_end_x-right_x))
                        mrg_s = (slice(left_z, -right_z if right_z else None), 
                                 slice(left_y, -right_y if right_y else None), 
                                 slice(left_x, -right_x if right_x else None))

                    elif len(n_tiles)==2:
                        src_s = (slice(src_start_y, src_end_y), 
                                 slice(src_start_x, src_end_x))
                        trg_s = (slice(src_start_y+left_y, src_end_y-right_y), 
                                 slice(src_start_x+left_x, src_end_x-right_x))
                        mrg_s = (slice(left_y, -right_y if right_y else None), 
                                 slice(left_x, -right_x if right_x else None))

                    yield src_s, trg_s, mrg_s


    volume_shape = source.shape
    if self.opts.proj_model:
        predict = np.zeros_like(source[0])
    else:
        predict = np.zeros_like(source)
    for src_s, trg_s, mrg_s in get_coord(volume_shape, patch_size, margin):
        input_fn = tf.estimator.inputs.numpy_input_fn(np.array([source[src_s]]), batch_size=1, shuffle=False)
        preds = list(transformer.predict(input_fn=input_fn))[0]
        predict[trg_s] = preds[mrg_s]

    return predict


