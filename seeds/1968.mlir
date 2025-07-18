module {
  func.func @main(%arg0: tensor<100x16x7x43x68xi8>) -> tensor<7x4678400xi8> {
    %r_0 = tosa.const_shape {values = dense<[ 27520, 238, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<100x16x7x43x68xi8>, !tosa.shape<3>) -> tensor<27520x238x5xi8>
    %r_1 = tosa.const_shape {values = dense<[ 7, 4678400 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<27520x238x5xi8>, !tosa.shape<2>) -> tensor<7x4678400xi8>
    return %1 : tensor<7x4678400xi8>
  }
}
