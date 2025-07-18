module {
  func.func @main(%arg0: tensor<13x11x29x63x14xi32>, %arg1: tensor<13x1x29x63x14xi32>, %arg2: tensor<38x63x82xi32>, %arg3: tensor<74x19x37x21x61x87xf32>, %arg4: tensor<15xi1>) -> (tensor<13x11x29x63x14xi1>, tensor<76x1x189xi32>, tensor<74x19x37x21x61x87xf32>, tensor<76x1xi32>, tensor<i32>, tensor<76x189xi32>, tensor<1xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<13x11x29x63x14xi32>, tensor<13x1x29x63x14xi32>) -> tensor<13x11x29x63x14xi1>
    %t_1 = tosa.const_shape {values = dense<[ 2, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %arg2, %t_1 : (tensor<38x63x82xi32>, !tosa.shape<3>) -> tensor<76x189x82xi32>
    %2 = tosa.tanh %arg3 : (tensor<74x19x37x21x61x87xf32>) -> tensor<74x19x37x21x61x87xf32>
    %3 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<76x189x82xi32>) -> tensor<76x189x1xi32>
    %4 = tosa.bitwise_and %3, %3 : (tensor<76x189x1xi32>, tensor<76x189x1xi32>) -> tensor<76x189x1xi32>
    %5 = tosa.identity %4 : (tensor<76x189x1xi32>) -> tensor<76x189x1xi32>
    %6 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %7 = tosa.transpose %5 {perms = array<i32: 0, 2, 1>} : (tensor<76x189x1xi32>) -> tensor<76x1x189xi32>
    %8 = tosa.logical_and %0, %0 : (tensor<13x11x29x63x14xi1>, tensor<13x11x29x63x14xi1>) -> tensor<13x11x29x63x14xi1>
    %9 = tosa.argmax %1 {axis = 2 : i32} : (tensor<76x189x82xi32>) -> tensor<76x189xi32>
    %10 = tosa.add %7, %7 : (tensor<76x1x189xi32>, tensor<76x1x189xi32>) -> tensor<76x1x189xi32>
    %11 = tosa.tanh %2 : (tensor<74x19x37x21x61x87xf32>) -> tensor<74x19x37x21x61x87xf32>
    %12 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<1xi1>
    %13 = tosa.reduce_min %9 {axis = 1 : i32} : (tensor<76x189xi32>) -> tensor<76x1xi32>
    %14 = tosa.reduce_all %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %15 = tosa.bitwise_not %9 : (tensor<76x189xi32>) -> tensor<76x189xi32>
    %16 = tosa.logical_left_shift %15, %15 : (tensor<76x189xi32>, tensor<76x189xi32>) -> tensor<76x189xi32>
    %17 = tosa.reverse %16 {axis = 0 : i32} : (tensor<76x189xi32>) -> tensor<76x189xi32>
    %18 = tosa.reduce_product %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %19 = tosa.argmax %14 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %20 = tosa.minimum %17, %17 : (tensor<76x189xi32>, tensor<76x189xi32>) -> tensor<76x189xi32>
    %21 = tosa.clamp %18 {min_val = 1 : i1, max_val = 1 : i1} : (tensor<1xi1>) -> tensor<1xi1>
    return %8, %10, %11, %13, %19, %20, %21 : tensor<13x11x29x63x14xi1>, tensor<76x1x189xi32>, tensor<74x19x37x21x61x87xf32>, tensor<76x1xi32>, tensor<i32>, tensor<76x189xi32>, tensor<1xi1>
  }
}
