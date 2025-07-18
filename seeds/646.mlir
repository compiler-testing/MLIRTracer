module {
  func.func @main(%arg0: tensor<8x58x59xi32>, %arg1: tensor<60xi1>, %arg2: tensor<81x75x9xf32>) -> (tensor<174x1xi32>, tensor<i32>, tensor<81x75x9xf32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<8x58x59xi32>) -> tensor<58x59xi32>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<58x59xi32>, tensor<58x59xi32>) -> tensor<58x59xi32>
    %2 = tosa.clamp %1 {min_val = -43 : i32, max_val = -24 : i32} : (tensor<58x59xi32>) -> tensor<58x59xi32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<58x59xi32>) -> tensor<58x1xi32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %3, %t_4 : (tensor<58x1xi32>, !tosa.shape<2>) -> tensor<174x1xi32>
    %5 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<60xi1>) -> tensor<1xi1>
    %6 = tosa.add %4, %4 : (tensor<174x1xi32>, tensor<174x1xi32>) -> tensor<174x1xi32>
    %7 = tosa.add %6, %4 : (tensor<174x1xi32>, tensor<174x1xi32>) -> tensor<174x1xi32>
    %8 = tosa.argmax %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %9 = tosa.exp %arg2 : (tensor<81x75x9xf32>) -> tensor<81x75x9xf32>
    return %7, %8, %9 : tensor<174x1xi32>, tensor<i32>, tensor<81x75x9xf32>
  }
}
