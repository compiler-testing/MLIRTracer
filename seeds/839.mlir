module {
  func.func @main(%arg0: tensor<5x63x10x11x66x26xf32>, %arg1: tensor<5x1x1x1x66x26xf32>, %arg2: tensor<78x66x74xi32>, %arg3: tensor<1x66x74xi32>, %arg4: tensor<24xi1>) -> (tensor<i1>, tensor<5x63x10x11x66x26xf32>, tensor<5x63x10x11x66x26xf32>, tensor<1xi32>, tensor<5x63x10x11x66x26xf32>, tensor<1xi1>, tensor<1x1xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<5x63x10x11x66x26xf32>, tensor<5x1x1x1x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %1 = tosa.ceil %0 : (tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<78x66x74xi32>, tensor<1x66x74xi32>) -> tensor<78x66x74xi32>
    %3 = tosa.add %1, %0 : (tensor<5x63x10x11x66x26xf32>, tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %4 = tosa.bitwise_not %2 : (tensor<78x66x74xi32>) -> tensor<78x66x74xi32>
    %5 = tosa.sub %2, %2 : (tensor<78x66x74xi32>, tensor<78x66x74xi32>) -> tensor<78x66x74xi32>
    %6 = tosa.maximum %4, %5 : (tensor<78x66x74xi32>, tensor<78x66x74xi32>) -> tensor<78x66x74xi32>
    %7 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<24xi1>) -> tensor<1xi1>
    %8 = tosa.argmax %7 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %9 = tosa.log %1 : (tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %10 = tosa.add %8, %8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %11 = tosa.reduce_all %7 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.pow %1, %1 : (tensor<5x63x10x11x66x26xf32>, tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %13 = tosa.reduce_sum %6 {axis = 2 : i32} : (tensor<78x66x74xi32>) -> tensor<78x66x1xi32>
    %r_14 = tosa.const_shape {values = dense<[ 5148 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %14 = tosa.reshape %13, %r_14 : (tensor<78x66x1xi32>, !tosa.shape<1>) -> tensor<5148xi32>
    %15 = tosa.greater %8, %10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %16 = tosa.reduce_product %14 {axis = 0 : i32} : (tensor<5148xi32>) -> tensor<1xi32>
    %17 = tosa.rsqrt %9 : (tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %18 = tosa.reduce_max %16 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %19 = tosa.bitwise_or %8, %10 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %r_20 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %20 = tosa.reshape %19, %r_20 : (tensor<i32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
    %21 = tosa.reduce_any %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %22 = tosa.sigmoid %12 : (tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %23 = tosa.reduce_product %18 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %24 = tosa.intdiv %23, %23 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %25 = tosa.argmax %20 {axis = 0 : i32} : (tensor<1x1x1xi32>) -> tensor<1x1xi32>
    %26 = tosa.rsqrt %3 : (tensor<5x63x10x11x66x26xf32>) -> tensor<5x63x10x11x66x26xf32>
    %27 = tosa.logical_or %21, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %28 = tosa.arithmetic_right_shift %25, %25 {round = false} : (tensor<1x1xi32>, tensor<1x1xi32>) -> tensor<1x1xi32>
    return %15, %17, %22, %24, %26, %27, %28 : tensor<i1>, tensor<5x63x10x11x66x26xf32>, tensor<5x63x10x11x66x26xf32>, tensor<1xi32>, tensor<5x63x10x11x66x26xf32>, tensor<1xi1>, tensor<1x1xi32>
  }
}
