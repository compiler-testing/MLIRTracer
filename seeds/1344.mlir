module {
  func.func @main(%arg0: tensor<14x58x22x76xf32>, %arg1: tensor<35x65x52x27x14xi32>, %arg2: tensor<1x1x52x1x1xi32>, %arg3: tensor<92x81x63x20x38xi1>, %arg4: tensor<1x81x1x20x1xi1>) -> (tensor<58x22x76xi32>, tensor<14x58x1x76xf32>, tensor<35x65x52x27x14xi32>, tensor<14x58x22x76xf32>, tensor<35x130x52x27x14xi32>, tensor<92x81x63x20x38xi1>, tensor<92x81x63x20x38xi1>) {
    %0 = tosa.abs %arg0 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %1 = tosa.identity %0 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %2 = tosa.sub %1, %1 : (tensor<14x58x22x76xf32>, tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %3 = tosa.floor %2 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %4 = tosa.ceil %3 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %5 = tosa.bitwise_xor %arg1, %arg2 : (tensor<35x65x52x27x14xi32>, tensor<1x1x52x1x1xi32>) -> tensor<35x65x52x27x14xi32>
    %6 = tosa.reduce_product %4 {axis = 2 : i32} : (tensor<14x58x22x76xf32>) -> tensor<14x58x1x76xf32>
    %7 = tosa.bitwise_not %5 : (tensor<35x65x52x27x14xi32>) -> tensor<35x65x52x27x14xi32>
    %8 = tosa.argmax %1 {axis = 0 : i32} : (tensor<14x58x22x76xf32>) -> tensor<58x22x76xi32>
    %9 = tosa.reciprocal %3 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %10 = tosa.bitwise_not %5 : (tensor<35x65x52x27x14xi32>) -> tensor<35x65x52x27x14xi32>
    %11 = tosa.clz %7 : (tensor<35x65x52x27x14xi32>) -> tensor<35x65x52x27x14xi32>
    %12 = tosa.concat %10, %11 {axis = 1 : i32} : (tensor<35x65x52x27x14xi32>, tensor<35x65x52x27x14xi32>) -> tensor<35x130x52x27x14xi32>
    %13 = tosa.reciprocal %6 : (tensor<14x58x1x76xf32>) -> tensor<14x58x1x76xf32>
    %14 = tosa.abs %12 : (tensor<35x130x52x27x14xi32>) -> tensor<35x130x52x27x14xi32>
    %15 = tosa.reverse %13 {axis = 0 : i32} : (tensor<14x58x1x76xf32>) -> tensor<14x58x1x76xf32>
    %16 = tosa.logical_and %arg3, %arg4 : (tensor<92x81x63x20x38xi1>, tensor<1x81x1x20x1xi1>) -> tensor<92x81x63x20x38xi1>
    %17 = tosa.logical_left_shift %11, %5 : (tensor<35x65x52x27x14xi32>, tensor<35x65x52x27x14xi32>) -> tensor<35x65x52x27x14xi32>
    %in_zp_18 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_18 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %18 = tosa.negate %17, %in_zp_18, %out_zp_18 : (tensor<35x65x52x27x14xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<35x65x52x27x14xi32>
    %19 = tosa.exp %9 : (tensor<14x58x22x76xf32>) -> tensor<14x58x22x76xf32>
    %20 = tosa.bitwise_and %14, %12 : (tensor<35x130x52x27x14xi32>, tensor<35x130x52x27x14xi32>) -> tensor<35x130x52x27x14xi32>
    %21 = tosa.logical_and %16, %16 : (tensor<92x81x63x20x38xi1>, tensor<92x81x63x20x38xi1>) -> tensor<92x81x63x20x38xi1>
    %22 = tosa.logical_not %16 : (tensor<92x81x63x20x38xi1>) -> tensor<92x81x63x20x38xi1>
    return %8, %15, %18, %19, %20, %21, %22 : tensor<58x22x76xi32>, tensor<14x58x1x76xf32>, tensor<35x65x52x27x14xi32>, tensor<14x58x22x76xf32>, tensor<35x130x52x27x14xi32>, tensor<92x81x63x20x38xi1>, tensor<92x81x63x20x38xi1>
  }
}
