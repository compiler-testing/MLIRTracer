module {
  func.func @main(%arg0: tensor<82xi8>, %arg1: tensor<82xi8>, %arg2: tensor<71x61x75x21x56x85xi64>, %arg3: tensor<1x1x1x1x56x85xi64>, %arg4: tensor<72x25x3x57xf32>, %arg5: tensor<1x25x3x57xf32>) -> (tensor<71x61x75x21x56x85xi1>, tensor<1xi1>, tensor<228x150x9xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<82xi8>, tensor<82xi8>) -> tensor<82xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<71x61x75x21x56x85xi64>, tensor<1x1x1x1x56x85xi64>) -> tensor<71x61x75x21x56x85xi1>
    %2 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<82xi1>) -> tensor<1xi1>
    %3 = tosa.bitwise_xor %1, %1 : (tensor<71x61x75x21x56x85xi1>, tensor<71x61x75x21x56x85xi1>) -> tensor<71x61x75x21x56x85xi1>
    %4 = tosa.pow %arg4, %arg5 : (tensor<72x25x3x57xf32>, tensor<1x25x3x57xf32>) -> tensor<72x25x3x57xf32>
    %5 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.equal %4, %4 : (tensor<72x25x3x57xf32>, tensor<72x25x3x57xf32>) -> tensor<72x25x3x57xi1>
    %9 = tosa.reverse %8 {axis = 3 : i32} : (tensor<72x25x3x57xi1>) -> tensor<72x25x3x57xi1>
    %r_10 = tosa.const_shape {values = dense<[ 228, 150, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.reshape %9, %r_10 : (tensor<72x25x3x57xi1>, !tosa.shape<3>) -> tensor<228x150x9xi1>
    return %3, %7, %10 : tensor<71x61x75x21x56x85xi1>, tensor<1xi1>, tensor<228x150x9xi1>
  }
}
