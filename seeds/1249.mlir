module {
  func.func @main(%arg0: tensor<5xf32>) -> (tensor<5xi1>, tensor<9xi1>) {
    %0 = tosa.abs %arg0 : (tensor<5xf32>) -> tensor<5xf32>
    %1 = tosa.equal %0, %0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %2 = tosa.greater_equal %0, %0 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %3 = tosa.logical_or %1, %1 : (tensor<5xi1>, tensor<5xi1>) -> tensor<5xi1>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<5xi1>) -> tensor<1xi1>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 9 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<9xi1>
    %7 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %8 = tosa.transpose %6 {perms = array<i32: 0>} : (tensor<9xi1>) -> tensor<9xi1>
    return %2, %8 : tensor<5xi1>, tensor<9xi1>
  }
}
