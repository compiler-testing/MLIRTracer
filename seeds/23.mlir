module {
  func.func @main(%arg0: tensor<31x59xi8>, %arg1: tensor<1x59xi8>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<35x68x12xf32>) -> (tensor<1x59xi1>, tensor<1x1xi1>, tensor<i32>, tensor<35x68x12xi1>, tensor<35x12x68xf32>, tensor<35x136x1xi1>, tensor<35x68x1xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<31x59xi8>, tensor<1x59xi8>) -> tensor<31x59xi1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<31x59xi1>) -> tensor<1x59xi1>
    %3 = tosa.bitwise_not %2 : (tensor<1x59xi1>) -> tensor<1x59xi1>
    %4 = tosa.reciprocal %arg4 : (tensor<35x68x12xf32>) -> tensor<35x68x12xf32>
    %5 = tosa.logical_or %3, %3 : (tensor<1x59xi1>, tensor<1x59xi1>) -> tensor<1x59xi1>
    %6 = tosa.intdiv %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<1x59xi1>) -> tensor<1x1xi1>
    %8 = tosa.sub %6, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %9 = tosa.equal %4, %4 : (tensor<35x68x12xf32>, tensor<35x68x12xf32>) -> tensor<35x68x12xi1>
    %10 = tosa.reverse %4 {axis = 2 : i32} : (tensor<35x68x12xf32>) -> tensor<35x68x12xf32>
    %11 = tosa.reverse %9 {axis = 0 : i32} : (tensor<35x68x12xi1>) -> tensor<35x68x12xi1>
    %12 = tosa.logical_right_shift %9, %9 : (tensor<35x68x12xi1>, tensor<35x68x12xi1>) -> tensor<35x68x12xi1>
    %13 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %14 = tosa.transpose %10 {perms = array<i32: 0, 2, 1>} : (tensor<35x68x12xf32>) -> tensor<35x12x68xf32>
    %15 = tosa.floor %14 : (tensor<35x12x68xf32>) -> tensor<35x12x68xf32>
    %16 = tosa.reduce_max %11 {axis = 2 : i32} : (tensor<35x68x12xi1>) -> tensor<35x68x1xi1>
    %17 = tosa.concat %16, %16 {axis = 1 : i32} : (tensor<35x68x1xi1>, tensor<35x68x1xi1>) -> tensor<35x136x1xi1>
    %18 = tosa.bitwise_and %16, %16 : (tensor<35x68x1xi1>, tensor<35x68x1xi1>) -> tensor<35x68x1xi1>
    return %5, %7, %8, %12, %15, %17, %18 : tensor<1x59xi1>, tensor<1x1xi1>, tensor<i32>, tensor<35x68x12xi1>, tensor<35x12x68xf32>, tensor<35x136x1xi1>, tensor<35x68x1xi1>
  }
}
