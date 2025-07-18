module {
  func.func @main(%arg0: tensor<68x10xf32>, %arg1: tensor<4x25x73x52x53xi64>, %arg2: tensor<1x25x73x52x1xi64>) -> (tensor<68x10xf32>, tensor<1x85x8xf32>, tensor<73x53x25x4x52xi64>, tensor<4x50x73x52x53xi1>, tensor<1x85x8xf32>) {
    %0 = tosa.ceil %arg0 : (tensor<68x10xf32>) -> tensor<68x10xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<4x25x73x52x53xi64>, tensor<1x25x73x52x1xi64>) -> tensor<4x25x73x52x53xi64>
    %r_2 = tosa.const_shape {values = dense<[ 1, 85, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %0, %r_2 : (tensor<68x10xf32>, !tosa.shape<3>) -> tensor<1x85x8xf32>
    %3 = tosa.clz %1 : (tensor<4x25x73x52x53xi64>) -> tensor<4x25x73x52x53xi64>
    %4 = tosa.log %0 : (tensor<68x10xf32>) -> tensor<68x10xf32>
    %5 = tosa.pow %2, %2 : (tensor<1x85x8xf32>, tensor<1x85x8xf32>) -> tensor<1x85x8xf32>
    %6 = tosa.equal %1, %3 : (tensor<4x25x73x52x53xi64>, tensor<4x25x73x52x53xi64>) -> tensor<4x25x73x52x53xi1>
    %7 = tosa.abs %3 : (tensor<4x25x73x52x53xi64>) -> tensor<4x25x73x52x53xi64>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %8 = tosa.negate %6, %in_zp_8, %out_zp_8 : (tensor<4x25x73x52x53xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<4x25x73x52x53xi1>
    %9 = "tosa.const"() {values = dense<[2, 4, 1, 0, 3]> : tensor<5xi32>} : () -> tensor<5xi32>
    %10 = tosa.transpose %7 {perms = array<i32: 2, 4, 1, 0, 3>} : (tensor<4x25x73x52x53xi64>) -> tensor<73x53x25x4x52xi64>
    %11 = tosa.bitwise_and %8, %8 : (tensor<4x25x73x52x53xi1>, tensor<4x25x73x52x53xi1>) -> tensor<4x25x73x52x53xi1>
    %12 = tosa.clz %11 : (tensor<4x25x73x52x53xi1>) -> tensor<4x25x73x52x53xi1>
    %13 = tosa.logical_or %12, %11 : (tensor<4x25x73x52x53xi1>, tensor<4x25x73x52x53xi1>) -> tensor<4x25x73x52x53xi1>
    %14 = tosa.concat %13, %6 {axis = 1 : i32} : (tensor<4x25x73x52x53xi1>, tensor<4x25x73x52x53xi1>) -> tensor<4x50x73x52x53xi1>
    %15 = tosa.exp %2 : (tensor<1x85x8xf32>) -> tensor<1x85x8xf32>
    return %4, %5, %10, %14, %15 : tensor<68x10xf32>, tensor<1x85x8xf32>, tensor<73x53x25x4x52xi64>, tensor<4x50x73x52x53xi1>, tensor<1x85x8xf32>
  }
}
