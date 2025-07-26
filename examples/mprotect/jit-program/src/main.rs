// To generate the .bin file, run:
// 1) cargo +succinct build --target riscv64im-succinct-zkvm-elf --release
// 2) riscv64-unknown-elf-objcopy -O binary ../../target/riscv64im-succinct-zkvm-elf/release/mprotect-jit-program mprotect-jit-program.bin

#![no_std]
#![no_main]

#[no_mangle]
pub extern "C" fn _start() -> u64 {
    let mut sum: u64 = 0;
    for i in 0..10 {
        sum += i;
    }
    sum
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
