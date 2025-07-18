module {
  func.func @main(%arg0: tensor<66x20x53xf32>, %arg1: tensor<66x1x1xf32>, %arg2: tensor<3x30x53xi16>, %arg3: tensor<3x1x1xi16>) -> (tensor<6x9x2xi16>, tensor<1x20x1xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<66x20x53xf32>, tensor<66x1x1xf32>) -> tensor<66x20x53xf32>
    %1 = tosa.logical_right_shift %arg2, %arg3 : (tensor<3x30x53xi16>, tensor<3x1x1xi16>) -> tensor<3x30x53xi16>
    %2 = tosa.exp %0 : (tensor<66x20x53xf32>) -> tensor<66x20x53xf32>
    %3 = tosa.sub %1, %1 : (tensor<3x30x53xi16>, tensor<3x30x53xi16>) -> tensor<3x30x53xi16>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<66x20x53xf32>) -> tensor<1x20x53xf32>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 2, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 6, 9, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %3, %s_5_start, %s_5_size : (tensor<3x30x53xi16>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<6x9x2xi16>
    %6 = tosa.reduce_product %4 {axis = 2 : i32} : (tensor<1x20x53xf32>) -> tensor<1x20x1xf32>
    return %5, %6 : tensor<6x9x2xi16>, tensor<1x20x1xf32>
  }
}
