module {
  func.func @main(%arg0: tensor<93x98x11x78x12x38xi32>, %arg1: tensor<1x1x1x78x1x1xi32>, %arg2: tensor<88x85x11x47xf32>) -> (tensor<93x98x11x78x12x38xi32>, tensor<1x85x11x1xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<93x98x11x78x12x38xi32>, tensor<1x1x1x78x1x1xi32>) -> tensor<93x98x11x78x12x38xi32>
    %1 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<88x85x11x47xf32>) -> tensor<1x85x11x47xf32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<93x98x11x78x12x38xi32>, tensor<93x98x11x78x12x38xi32>) -> tensor<93x98x11x78x12x38xi32>
    %3 = tosa.clamp %2 {min_val = -26 : i32, max_val = -17 : i32} : (tensor<93x98x11x78x12x38xi32>) -> tensor<93x98x11x78x12x38xi32>
    %4 = tosa.clz %3 : (tensor<93x98x11x78x12x38xi32>) -> tensor<93x98x11x78x12x38xi32>
    %5 = tosa.floor %1 : (tensor<1x85x11x47xf32>) -> tensor<1x85x11x47xf32>
    %6 = tosa.reduce_product %5 {axis = 3 : i32} : (tensor<1x85x11x47xf32>) -> tensor<1x85x11x1xf32>
    return %4, %6 : tensor<93x98x11x78x12x38xi32>, tensor<1x85x11x1xf32>
  }
}
