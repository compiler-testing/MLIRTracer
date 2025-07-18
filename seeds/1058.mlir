module {
  func.func @main(%arg0: tensor<42x40x88xi8>, %arg1: tensor<3x2xi32>, %arg2: tensor<f32>, %arg3: tensor<26x77x61xi1>) -> (tensor<42x40x88xi8>, tensor<1x77x61xi1>, tensor<1x1x1xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<42x40x88xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<42x40x88xi8>
    %1 = tosa.ceil %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<42x40x88xi8>, tensor<42x40x88xi8>) -> tensor<42x40x88xi8>
    %3 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<26x77x61xi1>) -> tensor<1x77x61xi1>
    %r_4 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %1, %r_4 : (tensor<f32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
    return %2, %3, %4 : tensor<42x40x88xi8>, tensor<1x77x61xi1>, tensor<1x1x1xf32>
  }
}
