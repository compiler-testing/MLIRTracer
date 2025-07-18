module {
  func.func @main(%arg0: tensor<79x95x22x10xi64>, %arg1: tensor<57x56x70xf32>) -> (tensor<1651100xi64>, tensor<1651100xi64>, tensor<57x56x70xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 1651100 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<79x95x22x10xi64>, !tosa.shape<1>) -> tensor<1651100xi64>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<1651100xi64>) -> tensor<1651100xi64>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1651100xi64>, tensor<1651100xi64>) -> tensor<1651100xi64>
    %4 = tosa.clz %3 : (tensor<1651100xi64>) -> tensor<1651100xi64>
    %5 = tosa.abs %4 : (tensor<1651100xi64>) -> tensor<1651100xi64>
    %6 = tosa.rsqrt %arg1 : (tensor<57x56x70xf32>) -> tensor<57x56x70xf32>
    %7 = tosa.add %5, %0 : (tensor<1651100xi64>, tensor<1651100xi64>) -> tensor<1651100xi64>
    %8 = tosa.sigmoid %6 : (tensor<57x56x70xf32>) -> tensor<57x56x70xf32>
    %9 = tosa.bitwise_and %4, %5 : (tensor<1651100xi64>, tensor<1651100xi64>) -> tensor<1651100xi64>
    %10 = tosa.greater_equal %6, %8 : (tensor<57x56x70xf32>, tensor<57x56x70xf32>) -> tensor<57x56x70xi1>
    return %7, %9, %10 : tensor<1651100xi64>, tensor<1651100xi64>, tensor<57x56x70xi1>
  }
}
