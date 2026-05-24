# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY NVIDIA CORPORATION AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Parallelization utilities.

Primary Usage:
- executor_generator: Runs a generator of functions in parallel using a process or thread pool.
- thread_generator: Runs a generator of functions in parallel using a thread pool.
- process_generator: Runs a generator of functions in parallel using a process pool.

Usage examples:

```python
from parallel import executor_generator

def func(i):
    # Heavy processing here
    return i

# Either use executor_generator (i.e. no parallelization), or use thread_generator or process_generator for parallelization.

for result in executor_generator((ProcessBound(func, i) for i in range(10)), max_queued=1):
    print(result)
```

"""

import concurrent.futures
from typing import (
    Callable,
    Generator,
    Generic,
    Literal,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    overload,
)

TRes = TypeVar("TRes")
P = ParamSpec("P")


class ProcessBound(Generic[P, TRes]):
    """Same as functools.partial, but allows usage in process pools."""

    def __init__(self, func: Callable[P, TRes], *args: P.args, **kwargs: P.kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs


@overload
def executor_generator(
    generator: Generator[Union[Callable[[], TRes], ProcessBound[P, TRes], TRes], None, None],
    max_queued: int = 1,
    executor: concurrent.futures.Executor | None = None,
    auto_raise: Literal[True] = True,
    in_order: bool = False,
) -> Generator[TRes, None, None]: ...


def executor_generator(
    generator: Generator[Union[Callable[[], TRes], ProcessBound[P, TRes], TRes], None, None],
    max_queued: int = 1,
    executor: concurrent.futures.Executor | None = None,
    auto_raise: bool = True,
    in_order: bool = False,
) -> Generator[Union[TRes, Exception], None, None]:
    """
    Runs a generator of functions in parallel using a process or thread pool.
    The generator should yield functions that take no arguments and return a result.
    The generator can also yield a non-callable, which will be yielded immediately.
    If the callable raises an exception, the exception will be yielded instead of the result.

    The generator will be exhausted when all functions have been called and all results have been yielded.

    Parameters:
        generator: The generator of functions to call
        max_queued: The maximum number of functions to queue up before waiting for results.
        executor: The executor to use. Must have been started. If None, just iterate over the generator in foreground.
        auto_raise: If True, exceptions raised by callables will automatically be reraised and will abort all processing. Otherwise exceptions will be returned as results.
        in_order: If True, the results will be yielded in the order of the generator (likely slower). Otherwise, the results will be yielded as they are completed.

    Returns:
        A generator of results (may be the exception of the generator)
    """
    if executor is None:
        # Just iterate over the generator.
        for item in generator:
            if isinstance(item, ProcessBound):
                try:
                    res = item.func(*item.args, **item.kwargs)
                except Exception as e:
                    if auto_raise:
                        raise e
                    yield e
                else:
                    yield res
            elif callable(item):
                try:
                    res = item()
                except Exception as e:
                    if auto_raise:
                        raise e
                    yield e
                else:
                    yield res
            else:
                yield item
    elif in_order:
        futures = []
        generator_iter = iter(generator)
        while True:
            for item in generator_iter:
                if isinstance(item, ProcessBound):
                    future = executor.submit(item.func, *item.args, **item.kwargs)
                    futures.append(future)
                    if len(futures) == max_queued:
                        break
                elif callable(item):
                    future = executor.submit(item)
                    futures.append(future)
                    if len(futures) == max_queued:
                        break
                else:
                    yield item
            else:
                # Generator finished
                break

            next_future = futures.pop(0)
            try:
                yield next_future.result()
            except Exception as e:
                if auto_raise:
                    # Abort all running tasks by cancelling remaining futures, abort iterating.
                    for f in futures:
                        f.cancel()
                    raise e
                yield e

        for future in futures:
            try:
                yield future.result()
            except Exception as e:
                if auto_raise:
                    # Abort all running tasks by cancelling remaining futures, abort iterating.
                    for f in futures:
                        f.cancel()
                    raise e
                yield e
    else:
        futures = set()
        generator_iter = iter(generator)
        while True:
            for item in generator_iter:
                if isinstance(item, ProcessBound):
                    future = executor.submit(item.func, *item.args, **item.kwargs)
                    futures.add(future)
                    if len(futures) == max_queued:
                        break
                elif callable(item):
                    future = executor.submit(item)
                    futures.add(future)
                    if len(futures) == max_queued:
                        break
                else:
                    yield item
            else:
                # Generator finished
                break
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    yield future.result()
                except Exception as e:
                    if auto_raise:
                        # Abort all running tasks by cancelling remaining futures, abort iterating.
                        for f in futures:
                            f.cancel()
                        raise e
                    yield e

        for future in concurrent.futures.as_completed(futures):
            try:
                yield future.result()
            except Exception as e:
                if auto_raise:
                    # Abort all running tasks by cancelling remaining futures, abort iterating.
                    for f in futures:
                        f.cancel()
                    raise e
                yield e


@overload
def thread_generator(
    generator: Generator[Union[Callable[[], TRes], ProcessBound[P, TRes], TRes], None, None],
    pool_size: int,
    max_queued: Optional[int] = None,
    auto_raise: Literal[True] = True,
    in_order: bool = False,
) -> Generator[TRes, None, None]: ...


def thread_generator(
    generator: Generator[Union[Callable[[], TRes], ProcessBound[P, TRes], TRes], None, None],
    pool_size: int,
    max_queued: Optional[int] = None,
    auto_raise: bool = True,
    in_order: bool = False,
) -> Generator[Union[TRes, Exception], None, None]:
    """
    Runs a generator of functions in parallel using a thread pool.
    The generator should yield functions that take no arguments and return a result.
    The generator can also yield a non-callable, which will be yielded immediately.
    If the callable raises an exception, the exception will be yielded instead of the result.

    The generator will be exhausted when all functions have been called and all results have been yielded.

    Parameters:
        generator: The generator of functions to call
        thread_pool_size: The number of threads to use
        max_queued: The maximum number of functions to queue up before waiting for results.
            If None, defaults to thread_pool_size.
        auto_raise: If True, exceptions raised by callables will automatically be reraised and will abort all processing. Otherwise exceptions will be returned as results.
        in_order: If True, the results will be yielded in the order of the generator (likely slower). Otherwise, the results will be yielded as they are completed.

    Returns:
        A generator of results (may be the exception of the generator)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        yield from executor_generator(
            generator,
            max_queued=max_queued or pool_size,
            executor=executor,
            auto_raise=auto_raise,
            in_order=in_order,
        )


@overload
def process_generator(
    generator: Generator[Union[Callable[[], TRes], ProcessBound[P, TRes], TRes], None, None],
    pool_size: int,
    max_queued: Optional[int] = None,
    auto_raise: Literal[True] = True,
    in_order: bool = False,
) -> Generator[TRes, None, None]: ...


def process_generator(
    generator: Generator[Union[ProcessBound[P, TRes], TRes], None, None],
    pool_size: int,
    max_queued: Optional[int] = None,
    auto_raise: bool = True,
    in_order: bool = False,
) -> Generator[Union[TRes, Exception], None, None]:
    """
    Runs a generator of functions in parallel using a process pool.
    The generator should yield functions that take no arguments and return a result.
    The generator can also yield a non-callable, which will be yielded immediately.
    If the callable raises an exception, the exception will be yielded instead of the result.

    The generator will be exhausted when all functions have been called and all results have been yielded.

    Parameters:
        generator: The generator of functions to call
        process_pool_size: The number of processes to use
        max_queued: The maximum number of functions to queue up before waiting for results.
            If None, defaults to process_pool_size.
        auto_raise: If True, exceptions raised by callables will automatically be reraised and will abort all processing. Otherwise exceptions will be returned as results.
        in_order: If True, the results will be yielded in the order of the generator (likely slower). Otherwise, the results will be yielded as they are completed.

    Returns:
        A generator of results (may be the exception of the generator)
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as executor:
        yield from executor_generator(
            generator,
            max_queued=max_queued or pool_size,
            executor=executor,
            auto_raise=auto_raise,
            in_order=in_order,
        )
