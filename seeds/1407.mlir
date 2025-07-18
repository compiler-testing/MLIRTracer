module {
  func.func @main(%arg0: tensor<33x59x85xi16>, %arg1: tensor<87x35x12x24x48x61xf32>) -> (tensor<935x177xi16>, tensor<87x35x12x24x48x61xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 165495 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<33x59x85xi16>, !tosa.shape<1>) -> tensor<165495xi16>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<165495xi16>) -> tensor<165495xi16>
    %2 = tosa.clamp %1 {min_val = -46 : i16, max_val = 80 : i16} : (tensor<165495xi16>) -> tensor<165495xi16>
    %r_3 = tosa.const_shape {values = dense<[ 935, 177 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<165495xi16>, !tosa.shape<2>) -> tensor<935x177xi16>
    %4 = tosa.ceil %arg1 : (tensor<87x35x12x24x48x61xf32>) -> tensor<87x35x12x24x48x61xf32>
    %5 = tosa.bitwise_or %3, %3 : (tensor<935x177xi16>, tensor<935x177xi16>) -> tensor<935x177xi16>
    %6 = tosa.minimum %4, %4 : (tensor<87x35x12x24x48x61xf32>, tensor<87x35x12x24x48x61xf32>) -> tensor<87x35x12x24x48x61xf32>
    return %5, %6 : tensor<935x177xi16>, tensor<87x35x12x24x48x61xf32>
  }
}
