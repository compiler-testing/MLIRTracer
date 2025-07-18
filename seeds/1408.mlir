module {
  func.func @main(%arg0: tensor<75x20x54x58xi8>, %arg1: tensor<4x2xi64>, %arg2: tensor<66x70x50x15xf32>, %arg3: tensor<42xi1>) -> (tensor<75x20x54x58xi8>, tensor<66x70x50x15xf32>, tensor<1xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<75x20x54x58xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<75x20x54x58xi8>
    %1 = tosa.bitwise_not %0 : (tensor<75x20x54x58xi8>) -> tensor<75x20x54x58xi8>
    %2 = tosa.tanh %arg2 : (tensor<66x70x50x15xf32>) -> tensor<66x70x50x15xf32>
    %3 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<42xi1>) -> tensor<1xi1>
    %4 = tosa.rsqrt %2 : (tensor<66x70x50x15xf32>) -> tensor<66x70x50x15xf32>
    %5 = tosa.arithmetic_right_shift %3, %3 {round = false} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %4, %5 : tensor<75x20x54x58xi8>, tensor<66x70x50x15xf32>, tensor<1xi1>
  }
}
