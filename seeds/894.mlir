module {
  func.func @main(%arg0: tensor<80xi64>, %arg1: tensor<35x78x62x56x85x91xf32>) -> (tensor<1xi1>, tensor<2xi1>, tensor<i32>, tensor<2xi1>, tensor<85x56x78x62x35x91xf32>, tensor<35x78x62x56x85x91xf32>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<80xi64>) -> tensor<1xi64>
    %1 = tosa.greater_equal %0, %0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.concat %2, %1 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %5 = tosa.greater %0, %0 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
    %6 = tosa.bitwise_not %4 : (tensor<2xi1>) -> tensor<2xi1>
    %7 = tosa.argmax %6 {axis = 0 : i32} : (tensor<2xi1>) -> tensor<i32>
    %8 = tosa.sub %7, %7 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.sigmoid %arg1 : (tensor<35x78x62x56x85x91xf32>) -> tensor<35x78x62x56x85x91xf32>
    %10 = tosa.logical_xor %3, %6 : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %11 = tosa.bitwise_or %8, %7 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = tosa.logical_not %3 : (tensor<2xi1>) -> tensor<2xi1>
    %13 = tosa.log %9 : (tensor<35x78x62x56x85x91xf32>) -> tensor<35x78x62x56x85x91xf32>
    %14 = tosa.maximum %13, %13 : (tensor<35x78x62x56x85x91xf32>, tensor<35x78x62x56x85x91xf32>) -> tensor<35x78x62x56x85x91xf32>
    %15 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %16 = tosa.transpose %9 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<35x78x62x56x85x91xf32>) -> tensor<85x56x78x62x35x91xf32>
    %17 = tosa.maximum %14, %13 : (tensor<35x78x62x56x85x91xf32>, tensor<35x78x62x56x85x91xf32>) -> tensor<35x78x62x56x85x91xf32>
    return %5, %10, %11, %12, %16, %17 : tensor<1xi1>, tensor<2xi1>, tensor<i32>, tensor<2xi1>, tensor<85x56x78x62x35x91xf32>, tensor<35x78x62x56x85x91xf32>
  }
}
