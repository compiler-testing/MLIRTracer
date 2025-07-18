module {
  func.func @main(%arg0: tensor<42x5x24x94x98xi8>, %arg1: tensor<5x2xi64>, %arg2: tensor<95x16x77xi8>) -> (tensor<42x5x24x94x98xi8>, tensor<16x77xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<42x5x24x94x98xi8>, !tosa.shape<10>, tensor<1xi8>) -> tensor<42x5x24x94x98xi8>
    %1 = tosa.argmax %arg2 {axis = 0 : i32} : (tensor<95x16x77xi8>) -> tensor<16x77xi32>
    return %0, %1 : tensor<42x5x24x94x98xi8>, tensor<16x77xi32>
  }
}
