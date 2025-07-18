module {
  func.func @main(%arg0: tensor<39xi8>, %arg1: tensor<39xi8>, %arg2: tensor<83x57x30x67x56x36xi64>, %arg3: tensor<83x1x1x67x1x1xi64>, %arg4: tensor<30x67x59x6x19xf32>) -> (tensor<1xi1>, tensor<83x57x30x67x56x36xi1>, tensor<30x67x59x6x19xf32>, tensor<1xi1>, tensor<30x67x59x6x19xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<39xi8>, tensor<39xi8>) -> tensor<39xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<83x57x30x67x56x36xi64>, tensor<83x1x1x67x1x1xi64>) -> tensor<83x57x30x67x56x36xi1>
    %2 = tosa.logical_not %0 : (tensor<39xi1>) -> tensor<39xi1>
    %3 = tosa.ceil %arg4 : (tensor<30x67x59x6x19xf32>) -> tensor<30x67x59x6x19xf32>
    %4 = tosa.clz %2 : (tensor<39xi1>) -> tensor<39xi1>
    %5 = tosa.concat %4, %2 {axis = 0 : i32} : (tensor<39xi1>, tensor<39xi1>) -> tensor<78xi1>
    %6 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<78xi1>) -> tensor<1xi1>
    %7 = tosa.equal %3, %3 : (tensor<30x67x59x6x19xf32>, tensor<30x67x59x6x19xf32>) -> tensor<30x67x59x6x19xi1>
    %8 = tosa.bitwise_or %1, %1 : (tensor<83x57x30x67x56x36xi1>, tensor<83x57x30x67x56x36xi1>) -> tensor<83x57x30x67x56x36xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %3, %in_zp_9, %out_zp_9 : (tensor<30x67x59x6x19xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<30x67x59x6x19xf32>
    %10 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<39xi1>) -> tensor<1xi1>
    %11 = tosa.clz %7 : (tensor<30x67x59x6x19xi1>) -> tensor<30x67x59x6x19xi1>
    return %6, %8, %9, %10, %11 : tensor<1xi1>, tensor<83x57x30x67x56x36xi1>, tensor<30x67x59x6x19xf32>, tensor<1xi1>, tensor<30x67x59x6x19xi1>
  }
}
